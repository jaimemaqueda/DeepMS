import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask

class LossPartE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dim_z = cfg.dim_z
        self.weights = cfg.loss_weights

    def forward(self, output):
        seq_z_tgt, seq_z_out = output["seq_z_tgt"], output["seq_z_out"]

        loss_seq_z_mse = F.mse_loss(seq_z_out, seq_z_tgt)

        loss_seq_z_cos = 1 - F.cosine_similarity(seq_z_out, seq_z_tgt, dim=-1).mean()

        loss_seq_z_mse = self.weights["loss_z_mse_weight"] * loss_seq_z_mse
        loss_seq_z_cos = self.weights["loss_z_cos_weight"] * loss_seq_z_cos

        res = {"loss_seq_z_mse": loss_seq_z_mse, "loss_seq_z_cos": loss_seq_z_cos}
        return res


class LossSeqAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_operations = cfg.n_operations
        self.dim_z = cfg.dim_z
        self.weights = cfg.loss_weights

    def forward(self, output):
        # Target & predictions
        seq_op_tgt, seq_zs_tgt = output["seq_op_tgt"], output["seq_zs_tgt"]
        seq_op_out, seq_zs_out = output["seq_op_out"], output["seq_zs_out"]
        
        # TODO: check CPU/GPU loading problems with the loss masks
        padding_mask_op = _get_padding_mask(seq_op_tgt, seq_dim=-2, extended=True)
        padding_mask_zs = _get_padding_mask(seq_op_tgt, seq_dim=-2, extended=False)

        loss_seq_op = F.cross_entropy(seq_op_out[padding_mask_op.bool()].reshape(-1, self.n_operations), seq_op_tgt[padding_mask_op.bool()].reshape(-1).long())
        loss_seq_zs_mse = F.mse_loss(seq_zs_out[padding_mask_zs.bool()].reshape(-1, self.dim_z), seq_zs_tgt[padding_mask_zs.bool()].reshape(-1, self.dim_z).float())
        loss_seq_zs_cos = 1 - F.cosine_similarity(seq_zs_out[padding_mask_zs.bool()].reshape(-1, self.dim_z), seq_zs_tgt[padding_mask_zs.bool()].reshape(-1, self.dim_z).float(), dim=-1).mean()

        loss_seq_op = self.weights["loss_seq_op_weight"] * loss_seq_op
        loss_seq_zs_mse = self.weights["loss_seq_zs_mse_weight"] * loss_seq_zs_mse
        loss_seq_zs_cos = self.weights["loss_seq_zs_cos_weight"] * loss_seq_zs_cos

        res = {"loss_seq_op": loss_seq_op, "loss_seq_zs_mse": loss_seq_zs_mse, "loss_seq_zs_cos": loss_seq_zs_cos}
        return res
    

class LossVoxAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.weights = cfg.loss_weights

    def forward(self, output):
        # Target & predictions
        voxel_tgt, voxel_out = output["voxel_tgt"], output["voxel_out"]

        # Calculate FPCE (False Positive Cross Entropy)
        loss_fpce = F.binary_cross_entropy(voxel_out[(voxel_tgt == 0).bool()], voxel_tgt[(voxel_tgt == 0).bool()])

        # Calculate FNCE (False Negative Cross Entropy)
        loss_fnce = F.binary_cross_entropy(voxel_out[(voxel_tgt == 1).bool()], voxel_tgt[(voxel_tgt == 1).bool()])

        # Apply weights
        loss_fpce = self.weights["loss_fpce_weight"] * loss_fpce
        loss_fnce = self.weights["loss_fnce_weight"] * loss_fnce

        res = {"loss_fpce": loss_fpce, "loss_fnce": loss_fnce}
        return res