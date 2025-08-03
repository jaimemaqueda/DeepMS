from .layers.voxel_encoding_decoding import *


class AutoEncoderVox(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderVox, self).__init__()

        self.vox_dim = cfg.vox_dim

        self.encoder_vox = VoxelEncoding(cfg.vox_dim, cfg.in_conv_channels, cfg.out_conv_channels, cfg.dim_z, cfg.n_conv_layers)
        self.decoder_vox = VoxelDecoding(cfg.vox_dim, cfg.out_conv_channels, cfg.in_conv_channels, cfg.dim_z, cfg.n_conv_layers)

    def forward(self, voxel,
                z=None, return_tgt=True, encode_mode=False):

        if z is None:
            z = self.encoder_vox(voxel)

        if encode_mode: return z # Shape [N, dim_z]

        voxel_logits = self.decoder_vox(z) # Shape [N, 1, vox_dim, vox_dim, vox_dim]

        res = {
            "voxel_out": voxel_logits # Shape [N, 1, vox_dim, vox_dim, vox_dim]
        }

        if return_tgt:
            res["voxel_tgt"] = voxel # Shape [N, 1, vox_dim, vox_dim, vox_dim]

        return res