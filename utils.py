import os, shutil, sys, importlib
import torch
from datetime import datetime
import os
from data import mnist_denormalize as denorm
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

def imread(fname, bounds=(-1, 1), **kwargs):
    from PIL import Image
    image = Image.open(fname, **kwargs).convert(mode='RGB')
    tensor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())), dtype=torch.float32)
    tensor = tensor.view(image.size[1], image.size[0], len(image.getbands()))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    vmin, vmax = bounds
    tensor = torch.clamp((vmax - vmin) * tensor + vmin, vmin, vmax)
    return tensor

def imwrite(image, fname, bounds=(0, 1), **kwargs):
    from PIL import Image
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    vmin, vmax = bounds
    # image = (image - vmin) / (vmax - vmin)
    image = (image * 255.0).round().clip(0, 255).to(torch.uint8)
    Image.fromarray(image.permute(1,2,0).cpu().numpy(), 'RGB').save(fname)



def find_latest_checkpoint(folder_path, exp_name=""):
    """
    Returns the latest checkpoint filename (based on modification time)
    in folder_path that contains exp_name in its filename.
    """
    if not os.path.exists(folder_path):
        return None
    files = [f for f in os.listdir(folder_path) if f.endswith(".pth") and exp_name in f]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
    return files[-1]


def handle_devices(device):
    if device == "ddp":
        use_ddp = True
        import torch.distributed as dist
        # DDP: assume launch with torchrun or similar so that LOCAL_RANK is set.
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        try:
            device = torch.device(device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise ValueError("CUDA is not available but a CUDA device was specified.")
        except Exception as e:
            raise ValueError(f"Invalid device specified: {device}") from e
        world_size = 1
        rank = 0
        use_ddp = False
    return use_ddp, device, world_size, rank



def create_experiment_dirs_ign(results_dir, dataset, exp_name="", save_code_snippet=True):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset}_{date_str}"
    if exp_name:
        base_name += f"_{exp_name}"
    exp_dir = os.path.join(results_dir, base_name)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    grid_dir = os.path.join(exp_dir, "generated")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    if save_code_snippet:
        code_dir = os.path.join(exp_dir, "code_snippet")
        os.makedirs(code_dir, exist_ok=True)

        # Ensure the new scripts can be found by importlib
        sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))

        src_paths = {
            "train_ign.py": os.path.abspath(sys.argv[0]),
            "lin_ign.py": importlib.import_module("lin_ign").__file__,
            "models.py": importlib.import_module("models").__file__,
            "utils.py": os.path.abspath(__file__),
            "data.py": importlib.import_module("data").__file__,
        }
        copied = []
        for name, src in src_paths.items():
            if src and os.path.isfile(src):
                shutil.copy2(src, os.path.join(code_dir, name))
                copied.append(name)
        print(f"[code_snippet] Saved {len(copied)} files to {code_dir}: {copied}")

    return exp_dir, ckpt_dir, grid_dir



def rk_coeffs(m, device):
    """Return   [1/0!, 1/1!, … , 1/m!]   as a 1-D tensor."""
    c = torch.empty(m + 1, device=device)
    c[0] = 1.
    for k in range(1, m + 1):
        c[k] = c[k - 1] / k
    return c


def poly_series(Z, coeffs):
    d = Z.shape[0]
    P   = coeffs[0] * torch.eye(d, dtype=Z.dtype, device=Z.device)
    Zk  = torch.eye(d, dtype=Z.dtype, device=Z.device)
    for k in range(1, len(coeffs)):
        Zk = Zk @ Z
        P  = P + coeffs[k] * Zk
    return P



def positional_encoding(coords, num_freqs, scale=1.0):
    freqs = scale * (2 ** torch.linspace(0, num_freqs - 1, num_freqs, device=coords.device))
    x = coords * freqs[None, :]  # [B, num_freqs]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)   # [B, num_freqs*2]





def make_ign_inputs(data_loader, size=32):
    import numpy as np
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    # ---- grab one real sample from the provided loader ----
    x_batch, *rest = next(iter(data_loader))   # [B,C,H,W]
    x0 = x_batch[0:1].detach().cpu()           # [1,C,H,W]

    inputs = OrderedDict()

    # 1) Gaussian noise
    for ind in range(6):
        inputs[f"gaussian_{ind}"] = torch.randn(1, 1, size, size)

    # 5) Noisy digit (from loader)
    inputs["digit_noisy"] = (x0 + 0.7 * torch.randn_like(x0))

    #
    jail = x0
    jail[:, :, :, ::7] = 1.
    inputs["digit_jail"] = jail

    # 6) Blurry digit (from loader)
    blur = T.GaussianBlur(kernel_size=9, sigma=2.)(x0)
    inputs["digit_blurry"] = blur

    # 4) Checkerboard
    board = torch.ones(1, 1, size, size)
    board[:, :, ::2, 1::2] = 0
    board[:, :, 1::2, ::2] = 0
    inputs["checker"] = board

    # 
    inputs["white"] = torch.ones(1, 1, size, size)

    # 2) Smiley face
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([size*0.20, size*0.20, size*0.40, size*0.40], fill=0)                # left eye
    d.ellipse([size*0.60, size*0.20, size*0.80, size*0.40], fill=0)                # right eye
    d.arc([size*0.20, size*0.45, size*0.80, size*0.85], 0, 180, fill=0, width=2)   # mouth
    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None, None, ...]
    inputs["smiley"] = 0.9 - arr  # invert to match MNIST fg

    # 3) ICLR text
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    d.text((2, size // 3), "ICLR", fill=0)
    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None, None, ...]
    inputs["iclr"] = 1.0 - arr






    return inputs


def make_celeba_inputs(data_loader, size=64, noise_sigma=0.5):
    """Build crafted/corrupted CelebA inputs for test-mode projection.

    The 3-channel counterpart of make_ign_inputs (which is MNIST/1-channel only).
    Returns an OrderedDict {name: tensor[1, 3, size, size]} with every tensor in
    the CelebA-NORMALIZED range (mean=std=0.5 → roughly [-1, 1]), i.e. exactly
    what the model's forward() expects. Display code should denormalize with
    data.celeba_denormalize before saving.

    Corruption families (each tests a different facet of "project onto the
    face manifold"):
      - real_*     : untouched test faces. Sanity check — f(real) ≈ real.
      - blur_*     : Gaussian-blurred. Tests deblurring.
      - noisy_*    : additive Gaussian noise. Tests denoising (the OOD eval task).
      - occluded_* : center square zeroed (gray). Tests inpainting.
      - lowres_*   : 4× down- then up-sampled. Tests super-resolution-like recovery.
      - noise_*    : pure Gaussian noise. Tests generation from scratch.

    The real source faces are pulled from the provided loader, so pass a test
    loader for held-out faces.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    x_batch, *rest = next(iter(data_loader))   # [B, 3, H, W], already normalized
    x_batch = x_batch[:, :3, :, :]             # guard against alpha
    inputs = OrderedDict()

    def src(i):
        # clone the i-th real face as a [1,3,size,size] tensor
        return x_batch[i:i + 1].clone()

    # 1) Clean real faces — f(real) should be near-identity.
    for i in range(3):
        inputs[f"real_{i}"] = src(i)

    # 2) Blurred.
    blur = T.GaussianBlur(kernel_size=9, sigma=3.0)
    for i in range(2):
        inputs[f"blur_{i}"] = blur(src(i))

    # 3) Additive Gaussian noise (clamped back to the normalized range).
    for i in range(2):
        inputs[f"noisy_{i}"] = (src(i) + noise_sigma * torch.randn(1, 3, size, size)).clamp(-1, 1)

    # 4) Center occlusion — zero (≈ gray after denorm) a central square.
    for i in range(2):
        occ = src(i)
        s = size // 4
        occ[:, :, s:3 * s, s:3 * s] = 0.0
        inputs[f"occluded_{i}"] = occ

    # 5) Low-res: downsample 4× then back up (loses high-frequency detail).
    for i in range(2):
        lr = F.interpolate(src(i), scale_factor=0.25, mode="bilinear", align_corners=False)
        lr = F.interpolate(lr, size=(size, size), mode="bilinear", align_corners=False)
        inputs[f"lowres_{i}"] = lr

    # 6) Pure Gaussian noise — generation from scratch.
    for i in range(3):
        inputs[f"noise_{i}"] = torch.randn(1, 3, size, size)

    return inputs


def plot_experiment_comparison(exp_dir1, exp_dir2, label1="CNN", label2="UNet"):
    df1 = pd.read_csv(os.path.join(exp_dir1, "metrics.csv"))
    df2 = pd.read_csv(os.path.join(exp_dir2, "metrics.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ['rec', 'sparse', 'tight']
    titles = ['Reconstruction Loss', 'Sparsity (A diag mean)', 'Tightness Loss']

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(df1['step'], df1[metric], label=label1, alpha=0.7)
        axes[i].plot(df2['step'], df2[metric], label=label2, alpha=0.7)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Global Step')
        axes[i].legend()      
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    plt.show()