import torch
import torch.optim as optim
from tqdm import tqdm
from model import AutoEncoderSeq
from .base import BaseTrainer
from .loss import LossSeqAE
from .scheduler import GradualWarmupScheduler
from mslib.macro import *


class TrainerSeqAE(BaseTrainer):
    def build_net(self, cfg):
        self.net = AutoEncoderSeq(cfg).cuda()
        if len(cfg.gpu_ids) > 1:
            self.net = torch.nn.DataParallel(self.net)

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        # TODO: check if the optimizer should be capturable
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.after_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100e3])
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step, after_scheduler=self.after_scheduler)

    def set_loss_function(self):
        self.loss_func = LossSeqAE(self.cfg).cuda()

    def forward(self, data):
        # TODO: check if the asynchronous data loading is necessary with non_blocking=True
        seq_op = data['seq_op'].cuda()  # (S, N)
        seq_zs = data['seq_zs'].cuda()  # (S, N, 256)

        outputs = self.net(seq_op, seq_zs)
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict

    def encode(self, data):
        """encode into latent vectors"""
        seq_op = data['seq_op'].cuda()  # (S, N)
        seq_zs = data['seq_zs'].cuda()  # (S, N, 256)
        z = self.net(seq_op, seq_zs, encode_mode=True)
        return z # (N, 256)

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs # {"seq_op_out": (S, N, 5), "seq_zs_out": (S, N, 256)}

    def logits2vec(self, outputs, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        seq_op_out = torch.argmax(torch.softmax(outputs['seq_op_out'], dim=-1), dim=-1)  # (S, N)
        seq_zs_out = outputs['seq_zs_out']  # (S, N, 256)

        if to_numpy:
            seq_op_out = seq_op_out.to(torch.uint8).detach().cpu().numpy()
            seq_zs_out = seq_zs_out.to(torch.float).detach().cpu().numpy()
        return {"seq_op_out": seq_op_out, "seq_zs_out": seq_zs_out}

    def evaluate(self, test_loader):
        """evaluation during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_rs_acc = []
        all_mill_acc = []
        all_drill_acc = []
        all_slant_acc = []
        all_eos_acc = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                seq_op = data['seq_op'].cuda()  # (S, N)
                seq_zs = data['seq_zs'].cuda()  # (S, N, 256)
                outputs = self.net(seq_op, seq_zs, return_tgt=False)
                outputs = self.logits2vec(outputs)
                seq_op_out = outputs["seq_op_out"] # (S, N)
                seq_op_gt = seq_op.to(torch.uint8).detach().cpu().numpy() # (S, N)

            rs_pos = np.where(seq_op_gt == RS_IDX)
            mill_pos = np.where(seq_op_gt == MILL_IDX)
            drill_pos = np.where(seq_op_gt == DRILL_IDX)
            slant_pos = np.where(seq_op_gt == SLANT_IDX)
            eos_pos = np.where((seq_op_gt == EOS_IDX).cumsum(axis=0) == 1)

            # Calculate accuracy for each operation
            seq_op_pred = (seq_op_out == seq_op_gt).astype(np.int)
            all_rs_acc.append(seq_op_pred[rs_pos])
            all_mill_acc.append(seq_op_pred[mill_pos])
            all_drill_acc.append(seq_op_pred[drill_pos])
            all_slant_acc.append(seq_op_pred[slant_pos])
            all_eos_acc.append(seq_op_pred[eos_pos])

        rs_acc = np.mean(np.concatenate(all_rs_acc))
        mill_acc = np.mean(np.concatenate(all_mill_acc))
        drill_acc = np.mean(np.concatenate(all_drill_acc))
        slant_acc = np.mean(np.concatenate(all_slant_acc))
        eos_acc = np.mean(np.concatenate(all_eos_acc))

        self.val_tb.add_scalars("seq_op_acc",
                                {"rs": rs_acc, "mill": mill_acc, "drill": drill_acc, "slant": slant_acc, "eos": eos_acc},
                                self.clock.epoch)

