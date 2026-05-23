import os, glob, random, torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math


def get_mnist_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # For MNIST: original size is typically 28; target size may differ.
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def mnist_denormalize(x):
    return x * 0.3081 + 0.1307

def mnist_normalize(x):
    # x in [0,1] -> normalized ((x - mean)/std)
    mean, std = 0.1307, 0.3081
    return (x - mean) / std


# Per-channel denormalize for CIFAR. x is [B, 3, H, W] in the normalized range.
# Multiplies and shifts per-channel to recover the [0, 1] pixel range used by
# downstream image-saving utilities (which clip to [0, 1]).
def _per_channel_denormalize(x, mean, std):
    import torch as _torch
    m = _torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = _torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * s + m


def cifar10_denormalize(x):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    return _per_channel_denormalize(x, mean, std)


def cifar100_denormalize(x):
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)
    return _per_channel_denormalize(x, mean, std)


def get_denormalize_fn(dataset_name):
    """Return the denormalize function appropriate for the dataset.

    Used by LinearIGN.valid() to convert model output back to [0, 1] for image
    saving. Falls back to mnist_denormalize for unknown datasets so existing
    runs don't break — but log a warning so the mismatch is visible.
    """
    if dataset_name == 'mnist':
        return mnist_denormalize
    if dataset_name == 'cifar10':
        return cifar10_denormalize
    if dataset_name == 'cifar100':
        return cifar100_denormalize
    if dataset_name == 'celeba':
        return celeba_denormalize
    print(f"[data] WARNING: no denormalize for dataset='{dataset_name}', using mnist_denormalize.")
    return mnist_denormalize


def get_normalize_stats(dataset_name):
    """Return the (mean, std) tuples used by the dataset's transforms.Normalize.

    Centralised so other code (notably ImageNetFeatureExtractor) can recover
    the IGN-normalized input range and convert it to whatever range a frozen
    pretrained model expects, without having to duplicate per-dataset constants.
    """
    if dataset_name == 'mnist':
        return (0.1307,), (0.3081,)
    if dataset_name == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if dataset_name == 'cifar100':
        return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    if dataset_name == 'celeba':
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    raise ValueError(
        f"No normalize stats registered for dataset '{dataset_name}'. "
        f"Add a branch to data.get_normalize_stats."
    )


def get_cifar10_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CIFAR-10 normalization values.
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def get_cifar100_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CIFAR-100 normalization statistics.
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def get_imagenet_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # Typical ImageNet normalization and transforms.
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize(int(target_size * 1.14)),
        transforms.RandomResizedCrop(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(target_size * 1.14)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.ImageNet(root='./data', split='train', download=False, transform=train_transform)
    val_dataset = datasets.ImageNet(root='./data', split='val', download=False, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=8, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

class CelebA_NoChecksum(datasets.CelebA):
    def _check_integrity(self) -> bool:
        # Only verify presence, not MD5
        base = os.path.join(self.root, "celeba")
        required = [
            "img_align_celeba",
            "list_attr_celeba.txt",
            "list_eval_partition.txt",
        ]
        ok = all(os.path.exists(os.path.join(base, r)) for r in required[:2]) and \
             os.path.isdir(os.path.join(base, required[0])) and \
             os.path.exists(os.path.join(base, required[2]))
        return ok


def celeba_denormalize(x):
    """x: [T,C,H,W] or [C,H,W] normalized with mean=std=(0.5,0.5,0.5) -> [0,1]."""
    import torch
    add_batch = (x.dim()==3)
    if add_batch:
        x = x.unsqueeze(0)
    mean = torch.tensor([0.5,0.5,0.5], dtype=x.dtype, device=x.device)[None,:,None,None]
    std  = torch.tensor([0.5,0.5,0.5], dtype=x.dtype, device=x.device)[None,:,None,None]
    y = x*std + mean
    y = y.clamp(0,1)
    return y[0] if add_batch else y



def get_celeba_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CelebA: using center crop and resize.
    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)
    # Standard CelebA-64 protocol: CenterCrop(178) first (drops most background
    # and keeps the head-centered framing the dataset is aligned for) then
    # resize to target_size. The earlier "Resize then CenterCrop" order
    # resized the short side to target before cropping, which produces a more
    # zoomed-out face at target=64.
    train_transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = CelebA_NoChecksum(root='./data', split='train', download=False, transform=train_transform, target_type="attr")
    val_dataset = CelebA_NoChecksum(root='./data', split='valid', download=False, transform=test_transform, target_type="attr")
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader



class MovingPoints(Dataset):
    def __init__(self, root, min_dt, max_dt, fps, ordered=False):
        self.tensors = [torch.load(p) for p in sorted(glob.glob(os.path.join(root, "*.pt")))]
        N, F = self.tensors[0].shape[-2], self.tensors[0].shape[-1]
        self.NF = N * F
        self.min_dt, self.max_dt, self.fps = float(min_dt), float(max_dt), float(fps)
        self.ordered = bool(ordered)

        # counts per file for ordered indexing: Bi * max(Ti-1, 1)
        self.counts = []
        for X in self.tensors:
            B, T = X.shape[0], X.shape[1]
            self.counts.append(B * max(T - 1, 1))
        self.cum = []
        s = 0
        for c in self.counts:
            self.cum.append(s)
            s += c
        self.total = s

    def __len__(self):
        return self.total if self.ordered else 10_000_000  # virtual length for random sampling

    def _locate(self, idx):
        # map idx → (fi, bi, t1_base) for ordered mode
        # each file contributes Bi*(T-1) slots; t1 will be adjusted after we know off
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cum[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        fi = lo
        X = self.tensors[fi]
        B, T = X.shape[0], X.shape[1]
        base = idx - self.cum[fi]
        span = max(T - 1, 1)
        bi = base // span
        t1_base = base % span
        bi = min(bi, B - 1)
        return fi, bi, t1_base

    def __getitem__(self, idx):
        if self.ordered:
            fi, bi, t1_base = self._locate(idx)
        else:
            fi = random.randrange(len(self.tensors))
            bi = random.randrange(self.tensors[fi].shape[0])

        X = self.tensors[fi]          # [B, T, N, F]
        T = X.shape[1]

        dt = random.uniform(self.min_dt, self.max_dt)
        off = int(round(dt * self.fps))
        if off < 1: off = 1
        if off >= T: off = T - 1
        dt = off / self.fps

        if self.ordered:
            t1 = min(t1_base, T - 1 - off)
        else:
            t1 = random.randint(0, T - 1 - off)

        x1 = X[bi, t1].reshape(self.NF) / 1000.
        x2 = X[bi, t1 + off].reshape(self.NF) / 1000.
        return x1, x2, torch.tensor(dt, dtype=X.dtype)


def get_moving_points_data_loaders(train_bs, val_bs, root, min_dt=0.1, max_dt=0.5, fps=30.0):
    train_dataset = MovingPoints(root + '/train', min_dt, max_dt, fps)
    val_dataset   = MovingPoints(root + '/valid', 5.0, 6.0, fps, ordered=True)

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=val_bs,   shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader



def get_data_loaders(dataset_name, train_bs, val_bs, orig_size=None, target_size=None, use_ddp=False, world_size=1, rank=0):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return get_mnist_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'cifar10':
        return get_cifar10_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'cifar100':
        return get_cifar100_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'imagenet':
        return get_imagenet_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'celeba':
        return get_celeba_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'cross':
        return get_moving_points_data_loaders(train_bs, val_bs, "./data/cross")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_cross_dataset(B, f_name):
    device = 'cpu'
    m = torch.rand(B, 1, 6, 1, dtype=torch.float32, device=device) * 1024.
    init_x_cm = torch.rand(B, 1, 1, 3, dtype=torch.float32, device=device) * 100 - 50
    init_ang = torch.rand(B, 1, 1, 1, dtype=torch.float32, device=device) * math.pi * 2
    init_v_cm = torch.rand(B, 1, 1, 3, dtype=torch.float32, device=device) * 100 - 50
    init_w = torch.rand(B, 1, 1, 3, dtype=torch.float32, device=device) * 2 - 1
    g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32, device=device).view(1,1,1,3)
    bounds = [-100, 100, -6000, 100, -100, 100]
    n_frames = 100
    T = 30
    pole_len = 1500

    x, v = generate_cross_stream_batch(pole_len, n_frames, T, m, 
                                       init_x_cm, init_ang, init_v_cm, 
                                       init_w, g, device)
    xvm = torch.cat([x, v, m.expand(B, n_frames, 6, 1)], -1)  # [B, n_frames, 6, 7]
    torch.save(xvm, f"./data/cross/{f_name}")
    print(f"{f_name} saved with {xvm.shape[0]} streams")

        


def generate_cross_stream_batch(pole_len, n_frames, t_max, m, init_x_cm, init_ang, init_v_cm, init_w, g, device):
    # calculate points initial positions
    r = pole_len / 2.0
    init_points = torch.tensor([
    [-r, 0.0, 0.0], [ r, 0.0, 0.0], [0.0, -r, 0.0], 
    [0.0,  r, 0.0], [0.0, 0.0, -r], [0.0, 0.0,  r]
    ], dtype=torch.float32, device=device).view(1, 1, 6, 3)

    # calculate center of mass
    init_cm_offset = ((m * init_points).sum(2, keepdim=True) / m.sum(2, keepdim=True))
    # init_x_cm = init_x_cm + init_cm_offset

    # set time schedule
    t = torch.linspace(0, t_max, n_frames, device=device).view(1, -1, 1, 1)

    # center of mass motion
    x_cm_t = 0.5 * g * t**2 + init_v_cm * t + init_x_cm  # [B, T, 1, 3]
    v_cm_t = g * t + init_v_cm  # [B, T, 1, 3]

    # rotational motion around CM
    eps = 1e-6

    # Normalize angular velocity to get a unit axis; use a default axis when ||w|| ~ 0
    w_norm = init_w.norm(dim=-1, keepdim=True)                 # [B,1,1,1]
    default_axis = torch.tensor([0.0, 0.0, 1.0], device=device).view(1,1,1,3)

    # normalized axis where possible
    axis_normed = init_w / torch.clamp(w_norm, min=eps)

    # choose normalized axis or default
    rot_ax = torch.where((w_norm > eps), axis_normed, default_axis)  # [B,1,1,3]
    # ensure exactly unit (guards against tiny numeric drift)
    rot_ax = rot_ax / rot_ax.norm(dim=-1, keepdim=True)
    angle_t = w_norm * t + init_ang

    # Rodrigues rotation formula
    points = (init_points - init_cm_offset)
    cos_t = torch.cos(angle_t)
    sin_t = torch.sin(angle_t)
    term1 = points * cos_t
    term2 = torch.cross(rot_ax, points, dim=3) * sin_t
    k_dot_v = torch.sum(rot_ax * points, dim=3, keepdim=True)
    term3 = rot_ax * k_dot_v * (1 - cos_t)
    x_wrt_cm = term1 + term2 + term3

    # World positions are the CM position plus the rotated points relative to CM
    x = x_wrt_cm + x_cm_t

    # Calculate local velocities
    v_wrt_cm = torch.cross(init_w, x_wrt_cm, dim=3)
    v = v_cm_t + v_wrt_cm

    return x, v



def visualize_cross(x, m, bounds):
    from collections import deque
    import numpy as np
    import vedo

    n_frames = x.shape[1]

    plt = vedo.Plotter(axes=1, bg='black',
                        title='Falling 3D Cross Dynamics (Randomized, Inertia Included)',
                        offscreen=True)

    plt.show(vedo.Box(bounds).alpha(0))
    video = vedo.Video("stream.mp4", duration=n_frames / 30.0, backend='ffmpeg')

    trail_len = 90  
    trails = [deque(maxlen=trail_len) for _ in range(6)]

    for t, x_t in enumerate(x[0, :].detach().cpu().numpy()):
        plt.clear()

        # update trails
        for j in range(6):
            trails[j].append(tuple(x_t[j]))

        # Draw three tubes for the 3D cross (points 0&1, 2&3, 4&5)
        pole1 = vedo.Tube(x_t[0:2], r=16.0, c='cyan').lighting('shiny')
        pole2 = vedo.Tube(x_t[2:4], r=16.0, c='cyan').lighting('shiny')
        pole3 = vedo.Tube(x_t[4:6], r=16.0, c='cyan').lighting('shiny')

        # Visualize points with radius proportional to mass
        point_actors = []
        for j in range(6):
            scaled_radius = 96.0 * (m[0, 0, j, 0].item() / 256.0)**0.5
            point_actor = vedo.Sphere(x_t[j], r=scaled_radius, c='yellow').lighting('metallic')
            point_actors.append(point_actor)

        trail_actors = []
        for j in range(6):
            pts = list(trails[j])
            if len(pts) > 2:
                # Taper the radius (thin tail -> thicker near head)
                rr = np.linspace(0.5, 5.0, num=len(pts)).tolist()
                # Slightly warm color, low alpha for elegance
                tr = vedo.Tube(pts, r=rr, c='yellow') #.alpha(0.1).lighting('off')
                trail_actors.append(tr)

        plt.add(pole1, pole2, pole3, *point_actors, *trail_actors)
        video.add_frame()
        print(f"\rAdded frame: {t+1}/{n_frames}", end="", flush=True)

    video.close()
    print("Animation saved to stream.mp4")


