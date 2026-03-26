import torch
import torch.nn as nn
from models import SpatialSplit2x2Rand, ActNorm2d

class UNetWrapper(nn.Module):
    """Wraps SongUNet to hide the noise_labels requirement."""
    def __init__(self, unet_creator, im_sz):
        super().__init__()
        # The split makes spatial size half (32 -> 16) and channels x2 (1 -> 2)
        self.unet = unet_creator(
            model_channels=64,     # Scaled down slightly to fit in memory easily
            in_channels=2,         # Expects 2 channels from the split
            out_channels=2,        # Outputs 2 channels
            img_resolution=im_sz,  # Resolution is 16 after split
            channel_mult=[1, 2, 2],
            num_blocks=2
        )

    def forward(self, x):
        # SongUNet requires noise labels; we provide dummy zeros
        B = x.shape[0]
        dummy_noise = torch.zeros(B, device=x.device)
        return self.unet(x, noise_labels=dummy_noise, class_labels=None)

class InvUnetBlock(nn.Module):
    """The Coupling Layer that makes the UNet invertible."""
    def __init__(self, unet_creator, im_sz):
        super().__init__()
        self.F = UNetWrapper(unet_creator, im_sz)
        self.G = UNetWrapper(unet_creator, im_sz)

    def forward(self, X_1, X_2):
        Y_1 = X_1 + self.F(X_2)
        Y_2 = X_2 + self.G(Y_1)
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = Y_2 - self.G(Y_1)
        X_1 = Y_1 - self.F(X_2)
        return X_1, X_2

class InvUnetV2(nn.Module):
    """The main Invertible UNet architecture to replace InvCNNNet."""
    def __init__(self, num_layers, in_channels, im_sz, unet_creator):
        super().__init__()
        split_im_sz = im_sz // 2  # 32 -> 16
        
        self.blocks = nn.ModuleList([InvUnetBlock(unet_creator, split_im_sz) for _ in range(num_layers)])
        self.splits = nn.ModuleList([SpatialSplit2x2Rand(chans=in_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([ActNorm2d(in_channels) for _ in range(num_layers)])

    def forward(self, X):
        for block, split, norm in zip(self.blocks, self.splits, self.norms):
            X_1, X_2 = split(X)
            X_1, X_2 = block(X_1, X_2)
            X = split.cat(X_1, X_2)
            X = norm(X)
        return X

    def inverse(self, Y):
        for block, split, norm in zip(reversed(self.blocks), reversed(self.splits), reversed(self.norms)):
            Y = norm.inverse(Y)
            Y_1, Y_2 = split.cat_inverse(Y)
            Y_1, Y_2 = block.inverse(Y_1, Y_2)
            Y = split.inverse(Y_1, Y_2)
        return Y