import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHA
from utils import positional_encoding
from torch.nn.utils import spectral_norm as sn



class Linearizer(nn.Module):
    def __init__(self, A_type, A_args, num_layers_x, split_type_x, split_args_x,
                 F_x, G_x, F_args_x=dict(), G_args_x=dict(),
                 num_layers_y=None, split_type_y=None, split_args_y=dict(), 
                 F_y=None, G_y=None, F_args_y=dict(), G_args_y=dict()):
        super().__init__()
        self.gx = InvNet(num_layers_x, split_type_x, split_args_x,
                         F_x, G_x, F_args_x, G_args_x)
        self.gy = (InvNet(num_layers_y, split_type_y, split_args_y,
                          F_y, G_y, F_args_y, G_args_y)
                   if F_y is not None else self.gx)
        self.A = A_type(**A_args)


    def forward(self, x, *A_args, **A_kwargs):
        g_x = self.gx(x)
        g_y_pred = self.A(g_x, *A_args, **A_kwargs)
        y_pred = self.gy.inverse(g_y_pred)
        return y_pred


class InvNet(nn.Module):
    def __init__(self, num_layers, split_type, split_args, F, G, F_args=None, G_args=None):
        super().__init__()
        self.blocks = nn.ModuleList([InvBlock(F, G, F_args, G_args) for _ in range(num_layers)])
        self.splits = nn.ModuleList([split_type(**split_args) for _ in range(num_layers)])

    def forward(self, X):
        for block, split in zip(self.blocks, self.splits):
            X_1, X_2 = split(X)
            X_1, X_2 = block(X_1, X_2)
            X = split.cat(X_1, X_2)
        return X

    def inverse(self, X):
        for block, split in zip(reversed(self.blocks), reversed(self.splits)):
            X_1, X_2 = split.cat_inverse(X)
            X_1, X_2 = block.inverse(X_1, X_2)
            X = split.inverse(X_1, X_2)
        return X



class InvBlock(nn.Module):
    def __init__(self, F, G, F_args=None, G_args=None):
        super().__init__()
        self.F = F(**F_args)
        self.G = G(**G_args)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2))
        Y_2 = (X_2 + self.G(Y_1))
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = (Y_2 - self.G(Y_1)) 
        X_1 = (Y_1 - self.F(X_2)) 
        return X_1, X_2



class InvertiblePermutationSplit(nn.Module):
    def __init__(self, n, axis=1):
        super(InvertiblePermutationSplit, self).__init__()
        perm = torch.randperm(n)
        self.register_buffer('perm', perm)
        inv_perm = torch.argsort(perm)
        self.register_buffer('inv_perm', inv_perm)
        self.axis = axis

    def forward(self, x):
        permuted = torch.index_select(x, self.axis, self.perm.to(x.device, non_blocking=True))
        return self.cat_inverse(permuted)

    def inverse(self, y1, y2):
        concat = self.cat(y1, y2)
        return torch.index_select(concat, self.axis, self.inv_perm)

    def cat(self, x1, x2):
        return torch.cat([x1, x2], self.axis)
        
    def cat_inverse(self, x):
        return x.chunk(2, self.axis)



class SpatialSplit2x2Rand(nn.Module):
    def __init__(self, chans=1):
        super().__init__()
        self.register_buffer('perm', torch.randperm(chans*4))

    def forward(self, x):  # x: (B,1,32,32)
        y = F.pixel_unshuffle(x, 2)          # (B,4,16,16)
        y = y[:, self.perm]                          # permute phases
        y1, y2 = y.chunk(2, axis=1)                      # (B,2,16,16) each
        return y1, y2

    def inverse(self, y1, y2):  # a,b: (B,2,16,16)
        y = torch.cat([y1, y2], dim=1)                      # (B,4,16,16)
        n = self.perm.numel()
        inv = torch.empty_like(self.perm)
        inv[self.perm] = torch.arange(n, device=self.perm.device)
        y = y[:, inv]                                     # undo perm
        x = F.pixel_shuffle(y, 2)            # (B,1,32,32)
        return x

    def cat(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        return F.pixel_shuffle(concat, 2)

    def cat_inverse(self, x):
        unshuffled = F.pixel_unshuffle(x, 2)
        return unshuffled.chunk(2, dim=1)


class ChannelSplit(nn.Module):
    """Pure channel-based split with a random per-layer permutation.

    Replaces SpatialSplit2x2Rand inside InvCNNNet. The key difference:
    no pixel_unshuffle/shuffle happens per layer — spatial structure is
    never decomposed into independent sub-grids, so the 4-quadrant artifact
    that appeared when generating from i.i.d. noise is eliminated.

    pixel_unshuffle(2) is applied ONCE at the InvCNNNet entry/exit instead,
    giving (B, 4C, H/2, W/2). All coupling layers then operate in this
    fixed downsampled space via channel splits only.

    forward(x)  → (x1, x2) : permute channels, chunk into two halves.
    inverse(x1, x2) → x    : cat halves, undo permutation.

    Invertibility proof (one layer):
      forward: X_perm = X[:, perm]; X1, X2 = chunk(X_perm)
      inverse: cat([X1_r, X2_r])[:, inv_perm]
             = X_perm[:, inv_perm] = X[:, perm][:, inv_perm] = X  ✓
    """
    def __init__(self, n_chans):
        super().__init__()
        perm = torch.randperm(n_chans)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))

    def forward(self, x):
        return x[:, self.perm].chunk(2, dim=1)

    def inverse(self, x1, x2):
        return torch.cat([x1, x2], dim=1)[:, self.inv_perm]



























class InvertiblePermutation(nn.Module):
    def __init__(self, n, axis=1):
        super(InvertiblePermutation, self).__init__()
        perm = torch.randperm(n)
        self.register_buffer('perm', perm)
        inv_perm = torch.argsort(perm)
        self.register_buffer('inv_perm', inv_perm)
        self.axis = axis

    def forward(self, x):
        return torch.index_select(x, self.axis, self.perm.to(x.device, non_blocking=True))

    def inverse(self, y):
        return torch.index_select(y, self.axis, self.inv_perm)




        

class InvTransformerNet(nn.Module):
    def __init__(self, num_heads, num_layers, patch_sz, im_sz, rgb=True):
        super().__init__()
        dim = 3*patch_sz**2 if rgb else patch_sz**2
        self.blocks = nn.ModuleList([InvTransformerBlock(dim//2, num_heads, patch_sz, im_sz) for _ in range(num_layers)])
        self.p = nn.ModuleList([InvertiblePermutation(dim, axis=2) for _ in range(num_layers)])
        self.unfold = nn.Unfold(kernel_size=patch_sz, stride=patch_sz)
        self.fold = nn.Fold(output_size=(im_sz, im_sz), kernel_size=patch_sz, stride=patch_sz)

    def forward(self, X):
        X = self.unfold(X).permute(0,2,1)
        for block, p in zip(self.blocks, self.p):
            X = p(X)
            X_1, X_2 = X.chunk(2, -1)
            X_1, X_2 = block(X_1, X_2)
            X = torch.cat([X_1, X_2], dim=-1)
        X = self.fold(X.permute(0,2,1))
        return X

    def inverse(self, Y):
        Y = self.unfold(Y).permute(0,2,1)
        for block, p in zip(reversed(self.blocks), reversed(self.p)):
            Y_1, Y_2 = Y.chunk(2, dim=-1)
            Y_1, Y_2 = block.inverse(Y_1, Y_2)
            Y = torch.cat([Y_1, Y_2], dim=-1)
            Y = p.inverse(Y)
        Y = self.fold(Y.permute(0,2,1))
        return Y


class InvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_sz=16, im_sz=256):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads)
        self.G = MLPSubblock(dim=dim, patch_sz=patch_sz, im_sz=im_sz)
        self.s = 1. #2 ** (-0.5)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2)) * self.s
        Y_2 = (X_2 + self.G(Y_1)) * self.s
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = (Y_2 / self.s - self.G(Y_1)) 
        X_1 = (Y_1 / self.s - self.F(X_2)) 
        return X_1, X_2


class MLPSubblock(nn.Module):
    def __init__(self, dim, patch_sz, im_sz, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            sn(nn.Linear(dim, dim * mlp_ratio, bias=True)),
            nn.GELU(),
            sn(nn.Linear(dim * mlp_ratio, dim, bias=True)))

    def forward(self, x):
        return self.norm2(self.mlp(self.norm1(x)))


class AttentionSubBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_ratio=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim * expand_ratio, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = MHA(dim * expand_ratio, num_heads, batch_first=True, bias=True)
        self.expand = sn(nn.Linear(dim, dim * expand_ratio, bias=True))
        self.shrink = sn(nn.Linear(dim * expand_ratio, dim, bias=True))
        self.v_start = dim * 2
        self.s = expand_ratio ** (-0.5)

    def forward(self, x):
        # we want biases for q, k but not for v
        # with torch.no_grad():
        #     self.attn.in_proj_bias[self.v_start:].zero_()
        #     self.attn.out_proj.bias.zero_()
        x = self.expand(x)
        x = self.norm1(x)

        x, _ = self.attn(x, x, x)
        x = self.shrink(x)
        x = self.norm2(x)
        return x





class InvCNNNet(nn.Module):
    def __init__(self, num_layers, im_sz, hidden_chans=128, data_channels=1):
        super().__init__()
        # Architecture: pixel_unshuffle(2) is applied ONCE at network entry/exit.
        # This maps (B, C, H, W) → (B, 4C, H/2, W/2) and all coupling layers
        # operate in this fixed downsampled space using ChannelSplit (channel-only
        # permutations, no spatial decomposition per layer).
        #
        # Previously SpatialSplit2x2Rand did pixel_unshuffle EVERY layer, which
        # decomposed the image into 4 spatially interlaced phases and processed
        # them semi-independently. For i.i.d. noise inputs the 4 phases were
        # uncorrelated, producing a visible 2x2 quadrant artifact in generated
        # samples. ChannelSplit eliminates this by never re-decomposing spatial
        # structure within the coupling stack.
        #
        # Coupling network shapes are identical to before:
        #   each CNNBlock sees (B, 2*data_channels, H/2, W/2).
        # ActNorm now operates on 4*data_channels channels (the full shuffled
        # representation) instead of data_channels.
        shuffled_chans = data_channels * 4   # channels in the pixel_unshuffle(2) space
        cnn_in_chans   = shuffled_chans // 2  # = 2 * data_channels, one coupling half

        self.blocks = nn.ModuleList([
            InvCNNBlock(im_sz // 2, hidden_chans=hidden_chans, cnn_in_chans=cnn_in_chans)
            for _ in range(num_layers)
        ])
        self.splits = nn.ModuleList([ChannelSplit(shuffled_chans) for _ in range(num_layers)])
        self.norms  = nn.ModuleList([ActNorm2d(shuffled_chans)    for _ in range(num_layers)])

    def forward(self, X):
        X = F.pixel_unshuffle(X, 2)              # (B, 4C, H/2, W/2) — enter once
        for block, split, norm in zip(self.blocks, self.splits, self.norms):
            X_1, X_2 = split(X)                  # permute channels + chunk
            X_1, X_2 = block(X_1, X_2)           # additive coupling
            X = split.inverse(X_1, X_2)          # cat + undo permutation
            X = norm(X)                           # per-channel ActNorm
        X = F.pixel_shuffle(X, 2)                # (B, C, H, W) — exit once
        return X

    def inverse(self, Y):
        Y = F.pixel_unshuffle(Y, 2)              # (B, 4C, H/2, W/2) — enter once
        for block, split, norm in zip(reversed(self.blocks), reversed(self.splits), reversed(self.norms)):
            Y = norm.inverse(Y)                  # undo ActNorm
            Y_1, Y_2 = split(Y)                  # same permute+chunk as forward
            Y_1, Y_2 = block.inverse(Y_1, Y_2)  # undo coupling
            Y = split.inverse(Y_1, Y_2)          # cat + undo permutation
        Y = F.pixel_shuffle(Y, 2)                # (B, C, H, W) — exit once
        return Y


#CNN V1 — parameterized on hidden_chans (the peak/3rd-down channel count).
# At hidden_chans=128 the pyramid is 2→8→32→128→512→128→32→8→2, exactly
# reproducing the previous hardcoded V1. Increase to 256 to roughly double
# width across the block (≈4× parameters at the peak — width scales params
# quadratically since both fan-in and fan-out grow). Reduce to 64 to halve.
# hidden_chans must be divisible by 16 since the periphery layers use
# hidden_chans // 16 channels.
class CNNBlock(nn.Module):
    def __init__(self, hidden_chans=128, in_chans=2, k_sz=4, bias=True):
        super().__init__()
        assert hidden_chans % 16 == 0, f"hidden_chans must be divisible by 16, got {hidden_chans}"
        h1 = hidden_chans // 16  # 8 at default
        h2 = hidden_chans // 4   # 32 at default
        h3 = hidden_chans        # 128 at default
        h4 = hidden_chans * 4    # 512 at default
        # Convolutions use kernel=4, stride=2, padding=1 — OVERLAPPING windows.
        # Previously kernel=2, stride=2, padding=0 (non-overlapping): each output
        # pixel saw a disjoint 2x2 input block, making the whole encoder a strict
        # quadtree. Information never crossed the coarsest (bottleneck) split, so
        # generation from i.i.d. noise produced independent image regions — the
        # 2x2 "four faces meeting at the center" artifact visible on CelebA
        # (64x64 → 32x32 block input → 2x2 bottleneck → 4 sealed quadrants).
        # MNIST was unaffected only because its 16x16 block input bottlenecks all
        # the way to 1x1 (full global mixing, no independent regions).
        #
        # kernel=4/stride=2/padding=1 produces IDENTICAL spatial sizes to the old
        # kernel=2/stride=2/padding=0 (down: 32→16→8→4→2; up: 2→4→8→16→32 — verified
        # for both Conv2d and ConvTranspose2d), so all downstream shapes and
        # invertibility are preserved. The only change is that each window now
        # overlaps its neighbours by 1 px per side, letting spatial information
        # cross quadrant boundaries and eliminating the artifact. Cost: 4x the
        # kernel weights per conv (4x4 vs 2x2).
        self.layer_1 = (nn.Conv2d(in_chans, h1, k_sz, 2, 1, bias=bias))
        self.layer_2 = (nn.Conv2d(h1, h2, k_sz, 2, 1, bias=bias))
        self.bn_1 = (nn.BatchNorm2d(h2))
        self.layer_3 = (nn.Conv2d(h2, h3, k_sz, 2, 1, bias=bias))
        self.layer_4 = (nn.Conv2d(h3, h4, k_sz, 2, 1, bias=bias))
        self.layer_5 = (nn.ConvTranspose2d(h4, h3, k_sz, 2, 1))
        self.layer_6 = (nn.ConvTranspose2d(h3, h2, k_sz, 2, 1))
        self.bn_2 = (nn.BatchNorm2d(h2))
        self.layer_7 = (nn.ConvTranspose2d(h2, h1, k_sz, 2, 1))
        self.layer_8 = (nn.ConvTranspose2d(h1, in_chans, k_sz, 2, 1))

    def forward(self, x):
        x1 = self.layer_1(x)
        x2 = self.layer_2(F.gelu(x1))
        # x2 = self.bn_1(x2)
        x3 = self.layer_3(F.gelu(x2))
        x4 = self.layer_4(F.gelu(x3))
        x5 = self.layer_5(F.gelu(x4))
        x6 = self.layer_6(F.gelu(x5))
        # x6 = self.bn_2(x6)
        x7 = self.layer_7(F.gelu(x6))
        x8 = self.layer_8(F.gelu(x7))
        return x8



# CNN V2
#class CNNBlock(nn.Module):
    #def __init__(self, hidden_chans, in_chans=2, k_sz=4, bias=True):
        #super().__init__()
        #self.layer_1 = (nn.Conv2d(in_chans, hidden_chans // 4, k_sz, 2, 1, bias=bias))
        #self.layer_2 = (nn.Conv2d(hidden_chans // 4, hidden_chans, k_sz, 2, 1, bias=bias))
        #self.bn_1 = (nn.BatchNorm2d(hidden_chans))        
        #self.layer_3 = (nn.Conv2d(hidden_chans, hidden_chans * 4, k_sz, 2, 1, bias=bias))
        #self.layer_4 = (nn.Conv2d(hidden_chans * 4, hidden_chans * 16, k_sz, 2, 1, bias=bias))
        #self.layer_5 = (nn.ConvTranspose2d(hidden_chans * 16, hidden_chans * 4, k_sz, 2, 1))
        #self.layer_6 = (nn.ConvTranspose2d(hidden_chans * 4, hidden_chans, k_sz, 2, 1))
        #self.bn_2 = (nn.BatchNorm2d(hidden_chans))
        #self.layer_7 = (nn.ConvTranspose2d(hidden_chans, hidden_chans // 4, k_sz, 2, 1))
        #self.layer_8 = (nn.ConvTranspose2d(hidden_chans // 4, in_chans, k_sz, 2, 1))
        
    #def forward(self, x):
        #x1 = self.layer_1(x) 
        #x2 = self.layer_2(F.gelu(x1))
        # #x2 = self.bn_1(x2)
        #x3 = self.layer_3(F.gelu(x2)) 
        #x4 = self.layer_4(F.gelu(x3))
        #x5 = self.layer_5(F.gelu(x4))
        #x6 = self.layer_6(F.gelu(x5))
        # #x6 = self.bn_2(x6)
        #x7 = self.layer_7(F.gelu(x6))
        #x8 = self.layer_8(F.gelu(x7))
        #return x8


class ActNorm2d(nn.Module):
    """Per-channel affine normalization with data-dependent init."""

    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, C, 1, 1))
        # 'initialized' is a registered buffer (not a plain bool) so it is saved
        # in the checkpoint and restored on load. Previously it was a Python bool,
        # which meant a loaded checkpoint always came back with initialized=False —
        # the first forward then re-ran the data-dependent _init and OVERWROTE the
        # trained bias/log_scale with statistics of whatever batch was passed
        # (e.g. corrupted test inputs in --mode test). As a buffer it round-trips
        # correctly; load_checkpoint also force-sets it True after any load.
        self.register_buffer('initialized', torch.tensor(False))
        self.eps = eps

    @torch.no_grad()
    def _init(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True) + self.eps
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std))
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            self._init(x)
        return (x + self.bias) * torch.exp(self.log_scale)

    def inverse(self, y):
        if not self.initialized:
            return y
        return y * torch.exp(-self.log_scale) - self.bias



class InvCNNBlock(nn.Module):
    def __init__(self, im_sz, hidden_chans=128, cnn_in_chans=2):
        super().__init__()
        # cnn_in_chans = channels per coupling-network half (one half of the
        # pixel_unshuffle(2) representation). For MNIST (1 input channel):
        # cnn_in_chans=2 (= 4*1//2). For CIFAR/CelebA (3): cnn_in_chans=6 (= 4*3//2).
        # Default 2 preserves the previous MNIST-only behaviour.
        self.F = CNNBlock(hidden_chans=hidden_chans, in_chans=cnn_in_chans)
        self.G = CNNBlock(hidden_chans=hidden_chans, in_chans=cnn_in_chans)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2))
        Y_2 = (X_2 + self.G(Y_1))
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = Y_2 - self.G(Y_1)
        X_1 = Y_1 - self.F(X_2)
        return X_1, X_2



def test_model_properties(model, bsz=64, test_y=False):
    device = next(model.parameters()).device
    im_sz_x = model.conf.im_shape[-1]
    n_inp_chans = 3 if model.rgb else 1
    x1, x2 = torch.randn(2*bsz, n_inp_chans, im_sz_x, im_sz_x, device=device).chunk(2, dim=0)
    a1, a2 = torch.randn(2*bsz, 1, 1, 1, device=device).chunk(2, dim=0)
    x = x1
    zx = torch.randn_like(x)
    if test_y:
        dim_y = model.dim_y
        y = torch.randn(bsz, 100, device=device)
        zy = torch.randn(bsz, dim_y, device=device)
    else:
        y = zy = None
    

    invertability_test(x, zx, y, zy, model, test_y)
    # linearity_test(x1, x2, a1, a2, model)
    unitarity_test(x, y, model, test_y)



@torch.no_grad()
def invertability_test(x, zx, y, zy, model, test_y, thr=1e-2):
    # X->Z->X
    zx_ = model.gx(x)
    x_ = model.gx.inverse(zx_)
    xzx = torch.norm(x - x_).item()
    xzx_ok = xzx < thr
    
    # Z->X->Z
    x_ = model.gx.inverse(zx)
    zx_ = model.gx(x_)
    zxz = torch.norm(zx - zx_).item()
    zxz_ok = zxz < thr

    if test_y:
        # Y->Z->Y
        zy_ = model.gy(y)
        y_ = model.gy.inverse(zy_)
        yzy = torch.norm(y - y_).item()
        yzy_ok = yzy < 1e-2

        # Z->Y->Z
        y_ = model.gy.inverse(zy)
        zy_ = model.gy(y_)
        zyz = torch.norm(zy - zy_).item()
        zyz_ok = zyz < thr

    print(f"X->Z->X: {xzx_ok} ({xzx})\nZ->X->Z: {zxz_ok} ({zxz})")
    if test_y:
        print(f"Y->Z->Y: {yzy_ok} ({yzy})\nZ->Y->Z: {zyz_ok} ({zyz})")


@torch.no_grad()
def linearity_test(x1, x2, a1, a2, model, thr=1e-4):
    # f(a1x1+a2x2)
    zx1, zx2 = model.gx(x1), model.gx(x2)
    zx_superpos = a1*zx1 + a2*zx2
    x_superpos = model.gx.inverse(zx_superpos)
    t = torch.randint(0, model.conf.T, (x1.size(0),), device=x1.device)
    f_superpos_x = model(x_superpos, t)

    # a1f(x1)+a2f(x2)
    y1, y2 = model(x1, t), model(x2, t)
    zy1, zy2 = model.gy(y1), model.gy(y2)
    zy_superpos = a1*zy1 + a2*zy2
    superpos_f_x = model.gy.inverse(zy_superpos)

    linearity_dist = (f_superpos_x - superpos_f_x).abs().mean()
    linearity_ok = linearity_dist < thr

    print(f"Linearity test: {linearity_ok} ({linearity_dist.item()})")
    
    zero_x = model.gx(x1*0).abs().mean()
    zero_x_ok = zero_x < thr
    zero_y = model.gy(y1*0).abs().mean()
    zero_y_ok = zero_y < thr

    print(f"Zero input test: X:{zero_x_ok} ({zero_x.item()}),   Y:{zero_y_ok} ({zero_y.item()})")


@torch.no_grad()
def unitarity_test(x, y, model, test_y, thr=10):
    zx_norm = model.gx(x).flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
    x_norm = x.flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
    ratio_x = (zx_norm / x_norm).mean()
    ratio_x_ok = ratio_x < thr

    if test_y:
        zy_norm = model.gy(y).flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
        ratio_y = (zy_norm / y_norm).mean()
        ratio_y_ok = ratio_y < thr

    print(f"Unitarity test: X:{ratio_x_ok} ({ratio_x.item()})")
    if test_y:
        print(f"Unitarity test: Y:{ratio_y_ok} ({ratio_y.item()})")


class GetMat(nn.Module):
    def __init__(self, im_sz, n_layers=5, hidden_dim=64):
        super().__init__()
        self.im_sz, self.hidden_dim = im_sz, hidden_dim
        sz = im_sz**2
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.final_layer = nn.Linear(hidden_dim, im_sz**4)
        self.register_buffer('base', torch.eye(sz)[None, ...])
        # self.base[:, sz:, sz:] = 0.


    def forward(self, t):
        sz = self.im_sz**2
        device = next(self.parameters()).device
        t = t * torch.ones(1, 1, device=device, dtype=torch.float32) # [B, 1] support both tensor and scalar
        mat = positional_encoding(t, num_freqs=self.hidden_dim//2) # [B, hidden_dim]
        for layer in self.layers:
            mat = F.gelu(layer(mat))
        mat = self.final_layer(mat) / self.im_sz
        return mat.view(-1, sz, sz) + self.base




class MatMul(nn.Module):
    def __init__(self, im_sz):
        super().__init__()
        self.im_sz = im_sz
        self.get_mat = GetMat(im_sz)

    def forward(self, x, t):
        A = self.get_mat(t)
        x_flat = x.view(x.shape[0], -1)
        return (torch.einsum("b t d, b d -> b t", A, x_flat)).view_as(x)

class SimpleMatMul(nn.Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        self.A = nn.Parameter(torch.randn(1, out_sz, in_sz)/ in_sz) 

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        return torch.einsum("b t d, b d -> b t", self.A, x_flat)




class ExponentDiagonal(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.diag = nn.Parameter(torch.ones(1, input_sz))

    def forward(self, x, t):
        evolver = (self.diag * t.unsqueeze(-1)).exp()
        return evolver * x

   

class InvMLPNet(nn.Module):
    def __init__(self, n_blocks, in_sz, hidden_sz, n_chunks=6):
        super().__init__()
        self.blocks = nn.ModuleList([InvMLPBlock(in_sz, hidden_sz) for _ in range(n_blocks)])
        self.p = nn.ModuleList([InvertibleMix(n_chunks) for _ in range(n_blocks)])

    def forward(self, X):
        X_1, X_2 = X.chunk(2, dim=-1)
        for block, p in zip(self.blocks, self.p):
            X_1, X_2 = p(X_1, X_2)
            X_1, X_2 = block(X_1, X_2)
        return torch.cat([X_1, X_2], dim=-1)

    def inverse(self, Y):
        Y_1, Y_2 = Y.chunk(2, dim=-1)
        for block, p in zip(reversed(self.blocks), reversed(self.p)):
            Y_1, Y_2 = block.inverse(Y_1, Y_2)
            Y_1, Y_2 = p.inverse(Y_1, Y_2)
        return torch.cat([Y_1, Y_2], dim=-1)


class InvMLPBlock(nn.Module):
    def __init__(self, in_sz, hidden_sz):
        super().__init__()
        self.F = MLPBlock(in_sz, hidden_sz)
        self.G = MLPBlock(in_sz, hidden_sz)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2))
        Y_2 = (X_2 + self.G(Y_1))
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = (Y_2 - self.G(Y_1)) 
        X_1 = (Y_1 - self.F(X_2)) 
        return X_1, X_2


class MLPBlock(nn.Module):
    def __init__(self, in_sz, hidden_sz):
        super().__init__()
        self.layer_1 = nn.Linear(in_sz//2, hidden_sz)
        self.act = nn.GELU()
        self.layer_2 = nn.Linear(hidden_sz, in_sz//2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        return x




class InvertibleMix(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = int(M)
        self.register_buffer("perm", torch.empty(0, dtype=torch.long))

    def forward(self, x1, x2):
        chunks = x1.chunk(self.M//2, dim=-1) + x2.chunk(self.M//2, dim=-1)
        self.perm = torch.randperm(self.M, device=x1.device)
        y1 = torch.cat([chunks[i] for i in self.perm[:self.M//2].tolist()], 1)
        y2 = torch.cat([chunks[i] for i in self.perm[self.M//2:].tolist()], 1)
        return y1, y2

    def inverse(self, y1, y2):
        C = 2 * y1.shape[1] // self.M
        y1c, y2c = list(y1.split(C, 1)), list(y2.split(C, 1))
        combo = [None] * self.M
        for k, i in enumerate(self.perm[:self.M//2].tolist()): combo[i] = y1c[k]
        for k, i in enumerate(self.perm[self.M//2:].tolist()): combo[i] = y2c[k]
        x1 = torch.cat(combo[:self.M//2], 1)
        x2 = torch.cat(combo[self.M//2:], 1)
        return x1, x2




class BasicLinearizer(nn.Module):
    def __init__(self, gx, A, gy=None):
        super().__init__()
        gy = gx if gy is None else gy
        self.gx, self.gy, self.A = gx, gy, A

    def forward(self, x, ret_intermid=False, *args, **kwargs):
        g_x = self.gx(x)
        g_y_pred = self.A(g_x, *args, **kwargs)
        y_pred = self.gy.inverse(g_y_pred)
        if ret_intermid:
            return y_pred, g_x, g_y_pred
        return y_pred



class RotationTrickEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_cont, beta=1.0):
        # 1. The Forward Pass: Snap to discrete binary mask (just like STE)
        x_disc = x_cont.round()

        # Save tensors for the backward pass
        ctx.save_for_backward(x_cont, x_disc)
        # beta controls the strength of the magnitude-scaling sparsity regularizer
        # applied in backward (see step 5). Stored as a plain attribute since it's
        # a scalar, not a tensor.
        ctx.beta = beta
        return x_disc

    @staticmethod
    def backward(ctx, grad_output):
        x_cont, x_disc = ctx.saved_tensors

        # 1. Calculate norms (with epsilon to prevent division by zero)
        eps = 1e-8
        norm_cont = x_cont.norm(dim=-1, keepdim=True).clamp_min(eps)
        norm_disc = x_disc.norm(dim=-1, keepdim=True).clamp_min(eps)

        # 2. Create unit vectors (u and v)
        u = x_cont / norm_cont
        v = x_disc / norm_disc

        # 3. Calculate cosine similarity between the vectors
        c = (u * v).sum(dim=-1, keepdim=True)

        # --- Handle Edge Cases ---
        # If the continuous vector was pushed to all zeros, or vectors are perfectly opposed,
        # rotation is undefined. Fallback safely to standard gradient.
        invalid_mask = (norm_disc <= eps) | (c <= -1.0 + eps)

        # 4. Compute the implicit R^T * grad_output using dot products
        # a = u \cdot g, b = v \cdot g
        a = (u * grad_output).sum(dim=-1, keepdim=True)
        b = (v * grad_output).sum(dim=-1, keepdim=True)

        # Algebraic expansion of the N-Dimensional rotation matrix transpose
        c_u = b + (b * c - a) / (1 + c)
        c_v = -a + (a * c - b) / (1 + c)

        grad_rotated = grad_output + (c_u * u) + (c_v * v)

        # 5. Apply the Scaling Factor, raised to the power beta.
        # The (norm_disc / norm_cont) factor is the implicit sparsity regularizer:
        # as the model grows A toward identity, norm_cont outgrows norm_disc, the
        # ratio (<=1) shrinks, the gradient shrinks, and A is held at a sparse
        # subspace. beta is a knob on that pressure:
        #   beta > 1  -> ratio^beta even smaller -> STRONGER sparsity -> lower A_active
        #   beta = 1  -> original behaviour (paper formulation)
        #   beta < 1  -> ratio^beta closer to 1 -> WEAKER sparsity  -> higher A_active
        #   beta = 0  -> no scaling at all -> A grows toward identity (STE-like)
        scale = (norm_disc / norm_cont).pow(ctx.beta)
        grad_input = scale * grad_rotated

        # Apply the fallback for invalid vectors (Dead Zero trap)
        grad_input = torch.where(invalid_mask, grad_output, grad_input)

        # Two inputs in forward (x_cont, beta); beta is a non-tensor hyperparameter
        # with no gradient, so return None for its slot.
        return grad_input, None



class IdempotentDiagonalOperator(nn.Module):
    def __init__(self, input_dim, binarizer='rotation', gumbel_tau=0.5, rotation_beta=1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, input_dim) - 2.)
        # binarizer ∈ {'rotation', 'ste', 'gumbel'} — which gradient estimator to use
        # for the round() forward pass. Default 'rotation' matches the pre-flag behaviour.
        self.binarizer = binarizer
        self.gumbel_tau = gumbel_tau
        # rotation_beta: exponent on the rotation trick's magnitude-scaling sparsity
        # regularizer. 1.0 = original. <1 = weaker sparsity (higher A_active),
        # >1 = stronger. Only affects the 'rotation' binarizer.
        self.rotation_beta = rotation_beta

    def forward(self, x, *args, **kwargs):
        # x shape: [B, C, H, W]
        original_shape = x.shape
        x_flat = x.view(x.shape[0], -1)
        probs = self.logits.sigmoid()
        self.probs = probs

        if self.binarizer == 'ste':
            # Plain straight-through estimator: forward = round, backward = identity.
            self.diag = probs.round().detach() + probs - probs.detach()
        elif self.binarizer == 'rotation':
            # Rotation trick (Fifty et al., 2024): rotates the gradient by the matrix
            # that aligns continuous probs vector to its rounded version.
            self.diag = RotationTrickEstimator.apply(probs, self.rotation_beta)
        elif self.binarizer == 'gumbel':
            # Gumbel-sigmoid + STE: noisy soft sample, then round, soft gradient via STE.
            if self.training:
                u = torch.rand_like(self.logits)
                gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            else:
                gumbel_noise = 0.0
            y_soft = torch.sigmoid((self.logits + gumbel_noise) / self.gumbel_tau)
            y_hard = y_soft.round()
            self.diag = y_hard.detach() - y_soft.detach() + y_soft
        else:
            raise ValueError(f"Unknown binarizer: {self.binarizer!r} (expected 'rotation', 'ste', or 'gumbel')")

        # Apply diagonal matrix A to the flattened latent vector g(x)
        y_flat = x_flat * self.diag

        return y_flat.view(original_shape)


class HouseholderRotation(nn.Module):
    """Parametrizes an orthogonal matrix Q ∈ SO(D) as a product of K Householder
    reflections of D-dimensional vectors.

    Q = H_{v_1} · H_{v_2} · ... · H_{v_K}
    where H_v = I - 2 v v^T / (v^T v) reflects across the hyperplane perpendicular
    to v. Each H_v is symmetric, orthogonal, and self-inverse. Products of
    Householders are exactly orthogonal regardless of the v values — no penalty
    or projection step required to maintain orthogonality.

    Cost per application: O(K · D) — K sequential rank-1 updates on the input.
    Storage:               K · D parameters (the K reflection vectors).

    For K = D, this parametrization spans the full orthogonal group SO(D)
    (Householder QR theorem). For K < D it spans a restricted but typically
    expressive subset — sufficient when Q only needs to align a few principal
    directions.

    Initialisation: vectors drawn from N(0, 1), which gives a random orthogonal
    Q at init. To start near identity, use a smaller scale (not done here —
    random init avoids trapping the optimizer near Q = I).
    """

    def __init__(self, D, K, eps=1e-8):
        super().__init__()
        self.D = D
        self.K = K
        self.eps = eps
        # K reflection vectors, each of dim D. Treated as plain learnable params.
        self.v = nn.Parameter(torch.randn(K, D))

    def _apply_sequence(self, x, order):
        """Apply Householder reflections to x in the given order.

        Each reflection: H_v x = x - 2 (v · x / v · v) v
        Implemented as one batched dot product + one scaled subtraction,
        avoiding ever forming the D×D Householder matrix.
        """
        for k in order:
            v_k = self.v[k]                                  # (D,)
            dot = x @ v_k                                    # (B,)
            denom = (v_k @ v_k).clamp(min=self.eps)          # scalar; eps guards div-by-zero
            x = x - (2.0 * dot / denom).unsqueeze(-1) * v_k  # (B, D)
        return x

    def forward(self, x):
        """Apply Q x. Input: (B, D), Output: (B, D)."""
        return self._apply_sequence(x, range(self.K))

    def inverse(self, x):
        """Apply Q^T x.

        Since Q = H_1 · H_2 · ... · H_K and each H_i is symmetric + self-inverse,
        Q^T = H_K^T · ... · H_1^T = H_K · ... · H_1. So we apply the same
        reflections in reverse order.
        """
        return self._apply_sequence(x, reversed(range(self.K)))


class IdempotentProjectionOperator(nn.Module):
    """A = Q L Q^T where Q is a learned orthogonal (via HouseholderRotation)
    and L = diag(b) is the same binary diagonal as IdempotentDiagonalOperator.

    Generalizes IdempotentDiagonalOperator: that operator is the special case
    Q = I (projection onto a coordinate-aligned K-dim subspace). With a learned
    Q, the projection direction is no longer constrained to coordinate axes —
    the model can rotate to align the K active dimensions with arbitrary
    subspaces of the input.

    Idempotence is structural:
      A² = Q L Q^T · Q L Q^T = Q L (Q^T Q) L Q^T = Q L · I · L Q^T = Q L² Q^T = Q L Q^T = A
    using L² = L (since L is binary) and Q^T Q = I (orthogonality of Q).

    Forward computation:
      1. Rotate into Q's frame:    x_rot = Q^T x       (using Q.inverse)
      2. Apply binary mask:         masked = x_rot * diag(b)
      3. Rotate back to input frame: y     = Q (masked) (using Q.forward)

    The binarization machinery for L (rotation trick, STE, gumbel-sigmoid) is
    identical to IdempotentDiagonalOperator — the same gradient estimators apply
    to L's probs regardless of Q.
    """

    def __init__(self, input_dim, n_householders=64,
                 binarizer='rotation', gumbel_tau=0.5, rotation_beta=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.binarizer = binarizer
        self.gumbel_tau = gumbel_tau
        # See IdempotentDiagonalOperator — exponent on the rotation trick's
        # implicit sparsity regularizer. Only affects the 'rotation' binarizer.
        self.rotation_beta = rotation_beta

        # L parametrization — identical to IdempotentDiagonalOperator
        self.logits = nn.Parameter(torch.randn(1, input_dim) - 2.)

        # Q parametrization. n_householders=0 → Q is identity (degenerate; the
        # operator then behaves identically to IdempotentDiagonalOperator).
        # Kept as an option for ablation experiments.
        self.use_rotation = (n_householders > 0)
        if self.use_rotation:
            self.Q = HouseholderRotation(input_dim, n_householders)
        else:
            self.Q = None

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x_flat = x.view(x.shape[0], -1)

        # Step 1: rotate into Q's frame.
        # Convention: A x = Q L Q^T x, so we apply Q^T first.
        if self.use_rotation:
            x_rot = self.Q.inverse(x_flat)
        else:
            x_rot = x_flat

        # Step 2: apply L — binarization identical to IdempotentDiagonalOperator.
        probs = self.logits.sigmoid()
        self.probs = probs

        if self.binarizer == 'ste':
            self.diag = probs.round().detach() + probs - probs.detach()
        elif self.binarizer == 'rotation':
            self.diag = RotationTrickEstimator.apply(probs, self.rotation_beta)
        elif self.binarizer == 'gumbel':
            if self.training:
                u = torch.rand_like(self.logits)
                gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            else:
                gumbel_noise = 0.0
            y_soft = torch.sigmoid((self.logits + gumbel_noise) / self.gumbel_tau)
            y_hard = y_soft.round()
            self.diag = y_hard.detach() - y_soft.detach() + y_soft
        else:
            raise ValueError(
                f"Unknown binarizer: {self.binarizer!r} (expected 'rotation', 'ste', or 'gumbel')"
            )

        masked = x_rot * self.diag

        # Step 3: rotate back to the input frame (apply Q).
        if self.use_rotation:
            y_flat = self.Q(masked)
        else:
            y_flat = masked

        return y_flat.view(original_shape)


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x
    def inverse(self, x, *args, **kwargs):
        return x
