import torch
import torch.nn as nn

def Norm(c):
    return nn.InstanceNorm2d(c, affine=True)

class Discriminator(nn.Module):
    """128×128 PatchGAN"""
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(Norm(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels * 2, 64, norm=False),  
            block(64, 128),                          
            block(128, 256),                         
            block(256, 512),                         
            nn.Conv2d(512, 1, 4, 1, 1),              
        )

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return self.model(x)
