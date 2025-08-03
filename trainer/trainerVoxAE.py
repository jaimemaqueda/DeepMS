import torch
import torch.optim as optim
from tqdm import tqdm
from model import AutoEncoderVox
from .base import BaseTrainer
from .loss import LossVoxAE
from .scheduler import GradualWarmupScheduler
from mslib.macro import *


class TrainerVoxAE(BaseTrainer):
    def build_net(self, cfg):
        self.net = AutoEncoderVox(cfg).cuda()
        if len(cfg.gpu_ids) > 1:
            self.net = torch.nn.DataParallel(self.net)

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        # TODO: check if the optimizer should be capturable
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = LossVoxAE(self.cfg).cuda()

    def forward(self, data):
        # TODO: check if the asynchronous data loading is necessary with non_blocking=True
        voxel = data['voxel'].cuda(non_blocking=True) # (N, 1, vox_dim, vox_dim, vox_dim)

        output = self.net(voxel)
        loss_dict = self.loss_func(output)

        return output, loss_dict

    def encode(self, data):
        """encode into latent vectors"""
        voxel = data['voxel'].cuda(non_blocking=True) # (N, 1, vox_dim, vox_dim, vox_dim)
        z = self.net(voxel, encode_mode=True)
        return z # (N, dim_z)

    def decode(self, z):
        """decode given latent vectors"""
        output = self.net(None, z=z, return_tgt=False)
        return output # {"voxel_out": (N, 1, vox_dim, vox_dim, vox_dim)}

    def binarize_voxel_logits(self, output, to_numpy=True):
        """binarize output voxel logits"""
        voxel_out = (output['voxel_out'] > 0.5)  # (N, 1, vox_dim, vox_dim, vox_dim)

        if to_numpy:
            voxel_out = voxel_out.to(torch.bool).detach().cpu().numpy()
        return voxel_out

    def evaluate(self, test_loader):
        """evaluation during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_rs_iou = []
        all_mill_iou = []
        all_drill_iou = []
        all_slant_iou = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                voxel = data['voxel'].cuda(non_blocking=True)  # (N, 1, vox_dim, vox_dim, vox_dim)
                output = self.net(voxel)
                voxel_out = self.binarize_voxel_logits(output, to_numpy=True)

            voxel_gt = voxel.detach().cpu().to(torch.bool).numpy() # (N, 1, vox_dim, vox_dim, vox_dim)
            operation_gt = data['operation'].to(torch.uint8).numpy()  # (N,)

            rs_pos = np.where(operation_gt == RS_IDX)
            mill_pos = np.where(operation_gt == MILL_IDX)
            drill_pos = np.where(operation_gt == DRILL_IDX)
            slant_pos = np.where(operation_gt == SLANT_IDX)

            # Calculate IoU for each voxel
            intersection = np.logical_and(voxel_gt, voxel_out)
            union = np.logical_or(voxel_gt, voxel_out)
            iou = np.sum(intersection, axis=(1, 2, 3, 4)) / np.sum(union, axis=(1, 2, 3, 4))

            # Append IoU to the respective lists
            if len(rs_pos[0]) > 0:
                all_rs_iou.append(iou[rs_pos])
            if len(mill_pos[0]) > 0:
                all_mill_iou.append(iou[mill_pos])
            if len(drill_pos[0]) > 0:
                all_drill_iou.append(iou[drill_pos])
            if len(slant_pos[0]) > 0:
                all_slant_iou.append(iou[slant_pos])
                
        rs_iou = np.mean(np.concatenate(all_rs_iou, axis=0))
        mill_iou = np.mean(np.concatenate(all_mill_iou, axis=0))
        drill_iou = np.mean(np.concatenate(all_drill_iou, axis=0))
        slant_iou = np.mean(np.concatenate(all_slant_iou, axis=0))

        self.val_tb.add_scalars("voxels_iou",
                                {"rs": rs_iou, "mill": mill_iou, "drill": drill_iou, "slant": slant_iou},
                                global_step=self.clock.epoch)
