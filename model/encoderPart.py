from .layers.voxel_encoding_decoding import *


class EncoderPart(nn.Module):
    def __init__(self, cfg):
        super(EncoderPart, self).__init__()

        self.n_conv_layers = cfg.n_conv_layers
        conv_channels = [cfg.in_conv_channels] + [cfg.out_conv_channels // (2 ** (cfg.n_conv_layers - i -1)) for i in range(cfg.n_conv_layers)]
        out_dimensions = cfg.vox_dim // (2 ** cfg.n_conv_layers)

        self.convs = nn.ModuleList()
        for i in range(self.n_conv_layers):
            self.convs.append(nn.Sequential(
            nn.Conv3d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv_channels[i + 1]),
            nn.LeakyReLU(0.2, inplace=True),
            ))

        self.fc = nn.Sequential(
            nn.Linear(cfg.out_conv_channels * out_dimensions * out_dimensions * out_dimensions, 4*cfg.dim_z),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4*cfg.dim_z),
            nn.Linear(4*cfg.dim_z, 2*cfg.dim_z),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(2*cfg.dim_z),
            nn.Linear(2*cfg.dim_z, cfg.dim_z),
            nn.Tanh()
        )

    def forward(self, fp_vox, seq_z_tgt = None):

        # fp_vox: (N, C, D, H, W)
        for conv in self.convs:
            fp_vox = conv(fp_vox)
        z = fp_vox.view(fp_vox.size(0), -1)  # Flatten
        z = self.fc(z)
        # z: (N, dim_z)

        res = {"seq_z_out": z}

        if seq_z_tgt is not None:
            res["seq_z_tgt"] = seq_z_tgt

        return res