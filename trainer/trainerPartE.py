import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
import os
import json
import h5py
from config import ConfigSeqAE, ConfigVoxAE
from trainer import TrainerVoxAE, TrainerSeqAE
from model import EncoderPart
from .base import BaseTrainer
from .loss import LossPartE
from .scheduler import GradualWarmupScheduler
from mslib.macro import *


class TrainerPartE(BaseTrainer):
    def build_net(self, cfg):
        self.net = EncoderPart(cfg).cuda()
        if len(cfg.gpu_ids) > 1:
            self.net = torch.nn.DataParallel(self.net)

        self.initialize_autoencoder_trainers(cfg)

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        # TODO: check if the optimizer should be capturable
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.after_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100e3])
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step, after_scheduler=self.after_scheduler)

    def set_loss_function(self):
        self.loss_func = LossPartE(self.cfg).cuda()

    def forward(self, data):
        # TODO: check if the asynchronous data loading is necessary with non_blocking=True
        fp_vox = data['fp_vox'].cuda(non_blocking = True)  # (N, 1, 128, 128, 128)
        seq_z_tgt = data['seq_z'].cuda(non_blocking = True)  # (N, 256)

        outputs = self.net(fp_vox, seq_z_tgt)
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict
    
    def initialize_autoencoder_trainers(self, cfg):
        with open(cfg.path_seq_ae_cfg, "r") as fp:
            cfg_seq_ae_dict = json.load(fp)
        with open(cfg.path_vox_ae_cfg, "r") as fp:
            cfg_vox_ae_dict = json.load(fp)

        # create experiment configuration containing all hyperparameters for SeqAE
        sys.argv = sys.argv[:1]
        for key, value in cfg_seq_ae_dict.items():
            sys.argv.append("--" + key)
            sys.argv.append(str(value))
        cfg_seq_ae = ConfigSeqAE('test')

        # create experiment configuration containing all hyperparameters for VoxAE
        sys.argv = sys.argv[:1]
        for key, value in cfg_vox_ae_dict.items():
            sys.argv.append("--" + key)
            sys.argv.append(str(value))
        cfg_vox_ae = ConfigVoxAE('test')

        # create training agents for SeqAE and VoxAE
        self.seq_ae_trainer = TrainerSeqAE(cfg_seq_ae)
        self.vox_ae_trainer = TrainerVoxAE(cfg_vox_ae)

        # load trained models from checkpoints
        self.seq_ae_trainer.load_ckpt(cfg_seq_ae.ckpt)
        self.seq_ae_trainer.net.eval()
        self.vox_ae_trainer.load_ckpt(cfg_vox_ae.ckpt)
        self.vox_ae_trainer.net.eval()

        
        # load mean and std of voxel latent vectors from the h5 file
        with h5py.File(os.path.join(cfg.data_root, "zs_mean_std.h5"), "r") as fp:
            self.mean_zs = torch.tensor(fp["mean"][:], dtype=torch.float).cuda()
            self.std_zs = torch.tensor(fp["std"][:], dtype=torch.float).cuda()

    def decode_z(self, seq_z_out):
        # decode the latent vector into a sequence of operations and voxels
        with torch.no_grad():
            outputs = self.seq_ae_trainer.decode(seq_z_out) # {"seq_op_out": (MAX_S, 1, 5), "seq_zs_out": (MAX_S, 1, 256)}
            seq_op_out = self.seq_ae_trainer.logits2vec(outputs)["seq_op_out"].squeeze() # (MAX_S,)
            seq_zs_out = outputs["seq_zs_out"].squeeze() * self.std_zs + self.mean_zs # (MAX_S, 256)

            eos_index = np.where(seq_op_out == EOS_IDX)[0]
            if len(eos_index) > 0:
                seq_op_out = seq_op_out[:eos_index[0]] # (S, )
                seq_zs_out = seq_zs_out[:eos_index[0]] # (S, 256)

            seq_vox_out = self.vox_ae_trainer.decode(seq_zs_out) # (S, 1, 128, 128, 128)
            seq_vox_out = self.vox_ae_trainer.binarize_voxel_logits(seq_vox_out) # (S, 1, 128, 128, 128)

        
        return seq_op_out, seq_vox_out


    def evaluate(self, test_loader):
        """evaluation during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_rs_iou = []
        all_rs_acc = []
        all_mill_iou = []
        all_mill_acc = []
        all_drill_iou = []
        all_drill_acc = []
        all_slant_iou = []
        all_slant_acc = []
        all_fp_iou = []
        all_eos_acc = []
        all_sequences_acc = []

        for b, data in enumerate(pbar):
            with torch.no_grad():
                # forward pass of the full model
                fp_vox = data['fp_vox'].cuda(non_blocking=True)  # (N, 1, 128, 128, 128)
                output = self.net(fp_vox)
                batch_seq_z = output["seq_z_out"] # (N, 256)
                batch_size = batch_seq_z.shape[0]

            for i in range(batch_size):
                # decode the latent vector into a sequence of operations and voxels
                seq_z_out = batch_seq_z[i].unsqueeze(0) # (1, 256)
                seq_op_out, seq_vox_out = self.decode_z(seq_z_out) # (S, ), (S, 1, 128, 128, 128)

                # get ground truth of the sequence of operations and voxels
                data_id = data['data_id'][i]
                folder, filename = data_id.rsplit("/", 1)
                with h5py.File(os.path.join(self.cfg.data_root, "seq_h5", folder, filename + ".h5"), "r") as h5file:
                    seq_op_gt = h5file["operations"][:] # (S, )
                    seq_vox_gt = h5file["voxels"][:] # (S, 128, 128, 128)

                length = seq_op_out.shape[0]
                sequence_acc = int(length == seq_op_gt.shape[0] - 1) # 1 if the sequence length is correct, 0 otherwise (FP voxel is not included)
                all_eos_acc.append(sequence_acc)
                fp_vox_out = np.ones(seq_vox_gt[0].shape, dtype=bool)

                for j in range(length):
                    op_out = seq_op_out[j]
                    vox_out = seq_vox_out[j].squeeze()
                    fp_vox_out = np.logical_and(fp_vox_out, ~vox_out)

                    if op_out == RS_IDX:
                        vox_out = ~vox_out
                    if j < seq_op_gt.shape[0] - 1:
                        op_gt = seq_op_gt[j]
                        vox_gt = seq_vox_gt[j]

                        # calculate IoU for each voxel
                        intersection = np.logical_and(vox_gt, vox_out)
                        union = np.logical_or(vox_gt, vox_out)
                        iou = np.sum(intersection) / np.sum(union)

                        # calculate accuracy for each operation
                        acc = int(op_out == op_gt)
                        sequence_acc = sequence_acc * acc

                        if op_gt == RS_IDX:
                            all_rs_iou.append(iou)
                            all_rs_acc.append(acc)
                        if op_gt == MILL_IDX:
                            all_mill_iou.append(iou)
                            all_mill_acc.append(acc)
                        if op_gt == DRILL_IDX:
                            all_drill_iou.append(iou)
                            all_drill_acc.append(acc)
                        if op_gt == SLANT_IDX:
                            all_slant_iou.append(iou)
                            all_slant_acc.append(acc)

                # calculate IoU for the final part
                fp_vox_gt = seq_vox_gt[-1]
                intersection = np.logical_and(fp_vox_gt, fp_vox_out)
                union = np.logical_or(fp_vox_gt, fp_vox_out)
                iou = np.sum(intersection) / np.sum(union)
                all_fp_iou.append(iou)
                all_sequences_acc.append(sequence_acc)

        
        # calculate mean IoU and accuracy for each operation
        rs_iou = np.mean(all_rs_iou)
        mill_iou = np.mean(all_mill_iou)
        drill_iou = np.mean(all_drill_iou)
        slant_iou = np.mean(all_slant_iou)
        fp_iou = np.mean(all_fp_iou)
        rs_acc = np.mean(all_rs_acc)
        mill_acc = np.mean(all_mill_acc)
        drill_acc = np.mean(all_drill_acc)
        slant_acc = np.mean(all_slant_acc)
        eos_acc = np.mean(all_eos_acc)
        sequence_acc = np.mean(all_sequences_acc)

        self.val_tb.add_scalars("seq_op_acc",
                                {"rs": rs_acc, "mill": mill_acc, "drill": drill_acc, "slant": slant_acc, "eos": eos_acc, "sequence": sequence_acc},
                                self.clock.epoch)
        
        self.val_tb.add_scalars("seq_vox_iou",
                                {"rs": rs_iou, "mill": mill_iou, "drill": drill_iou, "slant": slant_iou, "fp": fp_iou},
                                self.clock.epoch)





