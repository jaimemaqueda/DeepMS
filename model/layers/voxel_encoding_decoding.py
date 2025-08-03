import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelEncoding(nn.Module):
    def __init__(self, vox_dim, in_conv_channels, out_conv_channels, dim_z, n_conv_layers):
        super(VoxelEncoding, self).__init__()

        self.n_conv_layers = n_conv_layers
        conv_channels = [in_conv_channels] + [out_conv_channels // (2 ** (n_conv_layers - i -1)) for i in range(n_conv_layers)]
        out_dimensions = vox_dim // (2 ** n_conv_layers)

        self.convs = nn.ModuleList()
        for i in range(self.n_conv_layers):
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(conv_channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        self.fc = nn.Sequential(
            nn.Linear(out_conv_channels * out_dimensions * out_dimensions * out_dimensions, dim_z),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # x: (N, C, D, H, W)
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        # x: (N, dim_z)

        return x


class VoxelDecoding(nn.Module):
    def __init__(self, vox_dim, in_conv_channels, out_conv_channels, dim_z, n_conv_layers):
        super(VoxelDecoding, self).__init__()

        self.n_conv_layers = n_conv_layers
        self.in_conv_channels = in_conv_channels
        self.out_conv_channels = out_conv_channels
        self.vox_dim = vox_dim
        conv_channels = [in_conv_channels // (2 ** i) for i in range(n_conv_layers)] + [out_conv_channels]
        self.in_dimensions = vox_dim // (2 ** n_conv_layers)

        self.fc = nn.Sequential(
            nn.Linear(dim_z, in_conv_channels * self.in_dimensions * self.in_dimensions * self.in_dimensions),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconvs = nn.ModuleList()
        for i in range(self.n_conv_layers):
            if i == self.n_conv_layers - 1: # Last layer does not have batchnorm, activation, dropout
                self.deconvs.append(nn.Sequential(
                    nn.ConvTranspose3d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False)
                ))
            else:
                self.deconvs.append(nn.Sequential(
                    nn.ConvTranspose3d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(conv_channels[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True)
                ))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, dim_z)
        x = self.fc(x)
        x = x.view(-1, self.in_conv_channels, self.in_dimensions, self.in_dimensions, self.in_dimensions)
        for deconv in self.deconvs:
            x = deconv(x)
        x = self.sigmoid(x)
        # x: (N, C, D, H, W)
        return x
    
    

