import torch
import torch.nn as nn

# InstanceNorm helper
def Norm(c):
    return nn.InstanceNorm2d(c, affine=True)

class Down(nn.Module):
    """Downsampling block: Conv → (Norm) → LeakyReLU"""
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm:
            layers.append(Norm(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsampling block: ConvTranspose → Norm → ReLU → Skip Connection"""
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            Norm(out_c),
            nn.ReLU(True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class GeneratorUNet(nn.Module):
    """128×128 U-Net version (4 downs, 4 ups)"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # ------- Encoder -------
        self.d1 = Down(in_channels, 64, norm=False)   # 128 → 64
        self.d2 = Down(64, 128)                       # 64 → 32
        self.d3 = Down(128, 256)                      # 32 → 16
        self.d4 = Down(256, 512)                      # 16 → 8
        # Bottleneck
        self.bottleneck = Down(512, 512, norm=False)  # 8 → 4

        # ------- Decoder -------
        self.u1 = Up(512, 512, dropout=True)          # 4 → 8
        self.u2 = Up(512+512, 256, dropout=True)      # 8 → 16
        self.u3 = Up(256+256, 128)                    # 16 → 32
        self.u4 = Up(128+128, 64)                     # 32 → 64

        # Final upsample: 64+64 → output 128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64+64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.d1(x)   # 128→64
        d2 = self.d2(d1)  # 64→32
        d3 = self.d3(d2)  # 32→16
        d4 = self.d4(d3)  # 16→8
        b  = self.bottleneck(d4)  # 8→4

        # Decoder
        u1 = self.u1(b, d4)        # 4→8
        u2 = self.u2(u1, d3)       # 8→16
        u3 = self.u3(u2, d2)       # 16→32
        u4 = self.u4(u3, d1)       # 32→64

        return self.final(u4)      # 64→128
