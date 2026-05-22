"""Quantitative evaluation metrics for the IGN project.

Two families of metrics, both built around a small MNIST classifier:

1. Classifier-based generation metrics (compute on `f(z)` for random `z`):
   - entropy of mean class distribution → mode-collapse detector
   - mean per-sample confidence  → "do samples look like real digits?"
   - class coverage  → number of distinct digit classes the model produces

2. Out-of-distribution projection metrics (the paper's headline claim):
   - L1 between `f(corrupted_real)` and the clean original
   - classifier accuracy on `f(corrupted_real)` vs. ground-truth label

The classifier is a tiny CNN trained once via `python eval_metrics.py train ...`
and saved to disk. Training takes a few minutes; the trained `.pth` is then
loaded by `LinearIGN` at training-time validation.

Design notes:
- The classifier is trained on data passed through the same `get_data_loaders`
  pipeline as the IGN itself, so input range matches what the IGN produces.
- Inference is `torch.no_grad()` and uses ~256 samples per metric, cheap enough
  to run every validation epoch.
- All metrics are scalar floats / ints — easy to plot in WandB and compare
  across runs.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Classifier ------------------------------------------------------------

class MNISTClassifier(nn.Module):
    """Small CNN: ~100K params, reaches >99% test accuracy on standard MNIST.

    Designed to accept the same input shape as the IGN model: [B, 1, 32, 32]
    (resized MNIST) so it can be used as both a generation oracle (apply to
    `f(z)`) and a projection oracle (apply to `f(corrupted_real)`).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Input: [B, 1, 32, 32]
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # After three 2x2 pools: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        if return_features:
            return logits, feat
        return logits


class CIFAR10Classifier(nn.Module):
    """Small CNN for CIFAR-10 classification, used as a feature extractor / oracle.

    Same template as MNISTClassifier but with 3-channel input, wider conv channels,
    and a 256-dim feature layer + dropout to handle the harder dataset. Reaches
    ~78–82% test accuracy with ~15 epochs of training — modest by CIFAR standards
    but adequate for use as an oracle (entropy/coverage/confidence on generated
    samples) and as a perceptual feature extractor.

    Designed for the same input shape as the IGN model: [B, 3, 32, 32].
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Input: [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        # After three 2x2 pools: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        feat = F.relu(self.fc1(x))
        # Dropout only in training; perceptual-loss / eval calls happen with
        # the model in eval() mode so this is a no-op then.
        logits = self.fc2(self.dropout(feat))
        if return_features:
            return logits, feat
        return logits


class ImageNetFeatureExtractor(nn.Module):
    """Frozen ImageNet-pretrained ResNet18 used as a generic feature extractor.

    Drop-in replacement for the per-dataset classifiers when the dataset has
    no natural single-label target — currently CelebA. Provides the
    ``forward(x, return_features=True) -> (None, features)`` contract that
    ``_extract_features`` and the eval pipeline expect, but doesn't produce
    logits. The ``has_classifier = False`` flag tells the eval pipeline to
    skip the class-based metrics (gen_entropy/confidence/coverage,
    ood_class_acc/clean_acc) and emit only FID / PSNR / SSIM / L1.

    The IGN's input is per-dataset normalized (e.g. mean=std=0.5 for CelebA).
    ResNet18 expects ImageNet-normalized input (mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]). We precompute the affine that maps from one
    to the other so the forward pass costs a single multiply+add.

    Caveats:
      - Requires 3-channel input. Don't use with MNIST.
      - The ResNet was trained on 224x224 ImageNet; we feed it 64x64 CelebA
        crops. The conv stack still runs (it's fully convolutional up to
        avgpool) but features at this resolution are less informative than
        at native scale. Good enough for relative FID comparisons across
        IGN runs on the same dataset; not directly comparable to standard
        ImageNet-FID literature numbers.
    """
    has_classifier = False

    def __init__(self, input_mean, input_std):
        super().__init__()
        from torchvision import models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip the final fc; keep conv1 ... avgpool. Output: [B, 512, 1, 1].
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        imnet_mean = torch.tensor([0.485, 0.456, 0.406])
        imnet_std = torch.tensor([0.229, 0.224, 0.225])
        in_mean = torch.tensor(input_mean)
        in_std = torch.tensor(input_std)
        # ign_x = (real_x - in_mean) / in_std,   imnet_x = (real_x - imnet_mean) / imnet_std
        # =>  imnet_x = ign_x * (in_std / imnet_std) + (in_mean - imnet_mean) / imnet_std
        self.register_buffer('scale', (in_std / imnet_std).view(1, 3, 1, 1))
        self.register_buffer('shift', ((in_mean - imnet_mean) / imnet_std).view(1, 3, 1, 1))

    def forward(self, x, return_features=False):
        x = x * self.scale + self.shift
        feats = self.backbone(x).flatten(1)
        if return_features:
            return None, feats
        # The classifier-API contract returns logits when return_features is
        # False; we don't have any. Returning features is the safest fallback,
        # but callers using this extractor should pass return_features=True
        # and gate any logit-using code on .has_classifier.
        return feats


def make_classifier(dataset_name, num_classes=None):
    """Return the appropriate classifier class for a dataset, uninitialized.

    Both `eval_metrics.py train` (training the classifier from scratch) and
    `LinearIGN._maybe_load_eval` (loading the trained classifier for eval +
    perceptual loss) call this so the choice of architecture is centralized.
    """
    if dataset_name == 'mnist':
        return MNISTClassifier(num_classes=num_classes or 10)
    if dataset_name == 'cifar10':
        return CIFAR10Classifier(num_classes=num_classes or 10)
    if dataset_name == 'cifar100':
        return CIFAR10Classifier(num_classes=num_classes or 100)
    raise ValueError(
        f"No classifier defined for dataset '{dataset_name}'. "
        f"Supported: mnist, cifar10, cifar100."
    )


def train_classifier(model, train_loader, test_loader, device, n_epochs=5, lr=1e-3, log_every=200):
    """Train the given classifier model on the IGN's data pipeline.

    `model` is a freshly-constructed (and `.to(device)`'d) classifier instance
    — typically from `make_classifier(...)`. Returns the trained model.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            running_acc += (logits.argmax(-1) == y).float().sum().item()
            n += x.size(0)
            if step % log_every == 0:
                print(f"[clf train] epoch {epoch+1}/{n_epochs} step {step} "
                      f"loss={running_loss/n:.4f} acc={running_acc/n:.4f}")
        # End-of-epoch eval
        model.eval()
        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x).argmax(-1) == y).float().sum().item()
                test_n += x.size(0)
        print(f"[clf train] epoch {epoch+1} TEST acc={test_correct/test_n:.4f}")
    return model


# -- Distribution-level / image-quality metrics ----------------------------

@torch.no_grad()
def _extract_features(samples, classifier, batch_size=128):
    """Run classifier in feature-extraction mode over a batch (or larger set).

    Returns [N, D] features from the penultimate layer. Used by FID below.
    Batched in case the input doesn't fit in memory.
    """
    classifier.eval()
    feats = []
    for i in range(0, samples.shape[0], batch_size):
        chunk = samples[i:i + batch_size]
        _, f = classifier(chunk, return_features=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def compute_fid(real_features, fake_features, eps=1e-6):
    """Fréchet Inception Distance between two feature distributions.

    Computes the Wasserstein-2 distance between Gaussians fit to the two
    feature batches:
        FID = ||μ_r - μ_f||² + tr(Σ_r + Σ_f - 2·(Σ_r Σ_f)^{1/2})

    Lower is better. For MNIST/CIFAR with ~256 samples this is noisy but
    informative for relative comparison across runs. The standard "stable"
    FID uses 10K+ samples; for our internal-comparison use case 256-512 is
    enough to detect meaningful differences.

    Uses the project's local classifier as feature extractor instead of
    the InceptionV3 normally used for natural images — this is FID computed
    in the local feature space, not the standard ImageNet-FID.

    Args:
        real_features: [N, D] features from real images
        fake_features: [M, D] features from generated images
    Returns:
        scalar FID (float)
    """
    real_features = real_features.float()
    fake_features = fake_features.float()

    mu_r = real_features.mean(dim=0)
    mu_f = fake_features.mean(dim=0)

    # Covariance matrices, with small ridge for numerical stability
    D = real_features.shape[1]
    sigma_r = torch.cov(real_features.T) + eps * torch.eye(D, device=real_features.device)
    sigma_f = torch.cov(fake_features.T) + eps * torch.eye(D, device=real_features.device)

    diff = mu_r - mu_f
    mean_term = (diff * diff).sum()

    # Matrix sqrt of (Σ_r Σ_f) via eigendecomposition. The product isn't
    # symmetric but its eigenvalues are real and non-negative (the matrices
    # are PSD). Clamp at 0 for numerical safety.
    cov_prod = sigma_r @ sigma_f
    eigenvalues = torch.linalg.eigvals(cov_prod).real
    sqrt_trace = eigenvalues.clamp(min=0).sqrt().sum()

    trace_term = torch.trace(sigma_r) + torch.trace(sigma_f) - 2 * sqrt_trace

    fid = mean_term + trace_term
    return float(fid.item())


@torch.no_grad()
def compute_psnr(x, y, data_range=None):
    """Mean per-sample Peak Signal-to-Noise Ratio over a batch.

    PSNR = 20 · log10(data_range / sqrt(MSE))
    Higher is better. Reported in dB. Standard restoration-quality metric.

    Args:
        x, y: [B, C, H, W] tensors. Expected in matched range.
        data_range: max(x ∪ y) − min(x ∪ y). If None, inferred from x.
    Returns:
        mean PSNR across batch (float, in dB)
    """
    if data_range is None:
        data_range = float(x.max() - x.min())
    if data_range <= 0:
        data_range = 1.0
    mse = ((x - y) ** 2).mean(dim=[1, 2, 3])
    psnr = 20.0 * torch.log10(data_range / (mse.sqrt() + 1e-10))
    return float(psnr.mean().item())


def _gaussian_window(window_size, sigma, channels, device):
    """1D Gaussian window expanded to a 2D channel-wise filter for SSIM."""
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]  # [W, W]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def compute_ssim(x, y, window_size=11, sigma=1.5, data_range=None):
    """Mean per-sample Structural Similarity over a batch.

    SSIM = (2·μ_x·μ_y + c1) (2·σ_xy + c2) / ((μ_x² + μ_y² + c1) (σ_x² + σ_y² + c2))
    where μ, σ are Gaussian-windowed local statistics. Returns mean SSIM in [-1, 1]
    (typically [0, 1] for real images). Higher is better. Standard perceptual-quality
    metric for restoration.

    Args:
        x, y: [B, C, H, W] tensors in matched range
        window_size: Gaussian window size (must be odd)
        sigma:       Gaussian std for the window
        data_range:  max - min over data range; inferred from x if None
    Returns:
        mean SSIM across batch (float)
    """
    if data_range is None:
        data_range = float(x.max() - x.min())
    if data_range <= 0:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)
    pad = window_size // 2

    # Local means via Gaussian-weighted convolution, applied per-channel (groups=C)
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    ssim_map = (
        (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    )
    # Average over spatial + channel dims for per-sample SSIM, then over batch.
    return float(ssim_map.mean().item())


# -- Generation metrics ----------------------------------------------------

@torch.no_grad()
def classifier_metrics(samples, classifier, num_classes=10):
    """Apply classifier to generated samples, return entropy / confidence / coverage.

    Args:
        samples: [B, C, H, W] tensor in the same input range the classifier was
                 trained on (i.e. whatever `f(z)` produces — passed unchanged).
        classifier: a trained `MNISTClassifier` in eval mode.
        num_classes: 10 for MNIST.
    Returns:
        dict with three scalar entries:
          - gen_entropy    : entropy of the mean predicted-class distribution.
                             High (~log(num_classes)) = diverse; low = mode-collapsed.
          - gen_confidence : mean of per-sample max-softmax. High = digit-like.
          - gen_coverage   : number of distinct argmax classes seen. Max = num_classes.
    """
    classifier.eval()
    logits = classifier(samples)
    probs = F.softmax(logits, dim=-1)

    # Entropy of the average distribution (mode-collapse detector)
    mean_probs = probs.mean(0)
    entropy = -(mean_probs * (mean_probs.clamp(min=1e-8)).log()).sum().item()

    # Per-sample max softmax — "how confident is the classifier"
    confidence = probs.max(-1).values.mean().item()

    # Coverage: how many distinct classes appear as argmax across the batch
    preds = probs.argmax(-1)
    coverage = int(preds.unique().numel())

    return {
        'gen_entropy': float(entropy),
        'gen_confidence': float(confidence),
        'gen_coverage': coverage,
    }


# -- OOD projection metrics ------------------------------------------------

@torch.no_grad()
def ood_projection_metrics(ign_model, classifier, x_clean, y_true,
                            noise_sigma=0.3, compute_class_acc=True):
    """Project corrupted test images through `f`, measure projection quality.

    The headline claim of IGN is that `f` projects arbitrary inputs onto the
    learned data manifold. This function tests it on the cleanest version of
    that test: take real test images, add Gaussian noise of the given σ, push
    through `f`, and ask:
      (a) does the L1/PSNR/SSIM distance to the clean original improve after projection?
      (b) (if classifier has labels) does the classifier still recognise the original class?

    Args:
        ign_model: callable [B, C, H, W] → [B, C, H, W] (LinearIGN.forward).
        classifier: trained classifier OR a feature extractor. Only used here
                    if compute_class_acc=True (calls .forward to get logits).
        x_clean: clean test images, on the same device as ign_model.
        y_true:  ground-truth labels (LongTensor). Ignored if compute_class_acc=False.
        noise_sigma: Gaussian std added pre-projection.
        compute_class_acc: if False, skip the classifier-based accuracy metrics
                    (ood_class_acc, ood_clean_acc) — for datasets where the
                    "classifier" is really a feature-only extractor.
    Returns:
        dict with always-on keys: ood_{input,proj}_l1, ood_improvement,
        ood_{input,proj}_psnr, ood_psnr_improvement, ood_{input,proj}_ssim,
        ood_ssim_improvement. When compute_class_acc=True, also includes
        ood_class_acc and ood_clean_acc.
    """
    classifier.eval()
    z = torch.randn_like(x_clean)
    x_noisy = (x_clean + noise_sigma * z).clamp(-1, 1)

    fx_noisy = ign_model(x_noisy)

    input_l1 = F.l1_loss(x_noisy, x_clean).item()
    proj_l1 = F.l1_loss(fx_noisy, x_clean).item()

    # PSNR / SSIM on the projection vs clean ground truth.
    # data_range computed once from the clean batch so both input and
    # projection PSNRs use the same scale and can be directly compared.
    data_range = float(x_clean.max() - x_clean.min())
    if data_range <= 0:
        data_range = 1.0
    psnr_input = compute_psnr(x_noisy, x_clean, data_range=data_range)
    psnr_proj = compute_psnr(fx_noisy, x_clean, data_range=data_range)
    ssim_input = compute_ssim(x_noisy, x_clean, data_range=data_range)
    ssim_proj = compute_ssim(fx_noisy, x_clean, data_range=data_range)

    out = {
        'ood_input_l1': float(input_l1),
        'ood_proj_l1': float(proj_l1),
        'ood_improvement': float(input_l1 - proj_l1),
        'ood_input_psnr': float(psnr_input),
        'ood_proj_psnr': float(psnr_proj),
        'ood_psnr_improvement': float(psnr_proj - psnr_input),
        'ood_input_ssim': float(ssim_input),
        'ood_proj_ssim': float(ssim_proj),
        'ood_ssim_improvement': float(ssim_proj - ssim_input),
    }

    if compute_class_acc:
        proj_logits = classifier(fx_noisy)
        proj_acc = (proj_logits.argmax(-1) == y_true).float().mean().item()
        clean_logits = classifier(x_clean)
        clean_acc = (clean_logits.argmax(-1) == y_true).float().mean().item()
        out['ood_class_acc'] = float(proj_acc)
        out['ood_clean_acc'] = float(clean_acc)

    return out


# -- CLI for one-time classifier training ----------------------------------

def _cli_train(argv):
    """`python eval_metrics.py train --dataset {mnist,cifar10,cifar100} --out path.pth ...`

    Trains the appropriate classifier for the dataset (MNISTClassifier for mnist,
    CIFAR10Classifier for cifar10/cifar100). Saves the state_dict to --out for
    later loading by LinearIGN via --eval_classifier_path.
    """
    parser = argparse.ArgumentParser(
        description="Train the classifier used by IGN eval metrics + perceptual loss."
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Where to save the trained classifier weights. "
                             "Default: '<dataset>_classifier.pth' in the current dir.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs. Default depends on dataset: "
                             "5 for MNIST, 15 for CIFAR (harder).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--orig_im_size", type=int, default=None,
                        help="Default depends on dataset: 28 for MNIST, 32 for CIFAR.")
    parser.add_argument("--target_im_size", type=int, default=32,
                        help="Must match the size used in the IGN training pipeline.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    # Per-dataset sensible defaults — applied here rather than as argparse defaults
    # so we can vary based on --dataset selection.
    if args.out is None:
        args.out = f"{args.dataset}_classifier.pth"
    if args.epochs is None:
        args.epochs = 5 if args.dataset == "mnist" else 15
    if args.orig_im_size is None:
        args.orig_im_size = 28 if args.dataset == "mnist" else 32

    # Lazy import to avoid pulling torchvision at module import time
    from data import get_data_loaders

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = make_classifier(args.dataset)
    print(f"Training {type(model).__name__} for {args.dataset} on {device}, "
          f"{args.epochs} epochs, batch {args.batch_size}")

    train_loader, test_loader = get_data_loaders(
        args.dataset, args.batch_size, args.batch_size,
        args.orig_im_size, args.target_im_size,
    )

    model = train_classifier(model, train_loader, test_loader, device,
                              n_epochs=args.epochs, lr=args.lr)
    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.dataset} classifier weights to '{args.out}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_metrics.py train [--out path.pth ...]")
        sys.exit(1)
    sub = sys.argv[1]
    if sub == "train":
        _cli_train(sys.argv[2:])
    else:
        print(f"Unknown subcommand '{sub}'. Use 'train'.")
        sys.exit(1)
