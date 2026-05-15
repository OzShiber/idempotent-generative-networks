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
def ood_projection_metrics(ign_model, classifier, x_clean, y_true, noise_sigma=0.3):
    """Project corrupted test images through `f`, measure projection quality.

    The headline claim of IGN is that `f` projects arbitrary inputs onto the
    learned data manifold. This function tests it on the cleanest version of
    that test: take real test images, add Gaussian noise of the given σ, push
    through `f`, and ask:
      (a) does the L1 distance to the clean original go DOWN after projection?
      (b) does the classifier still recognise the original digit class?

    Args:
        ign_model: callable [B, C, H, W] → [B, C, H, W] (LinearIGN.forward).
        classifier: trained MNISTClassifier.
        x_clean: clean test images, on the same device as ign_model.
        y_true:  ground-truth labels (LongTensor).
        noise_sigma: Gaussian std added pre-projection.
    Returns:
        dict with:
          - ood_input_l1     : L1(x_noisy, x_clean) — how corrupted was the input
          - ood_proj_l1      : L1(f(x_noisy), x_clean) — how close is the projection
          - ood_improvement  : input_l1 - proj_l1 (positive = projection moved closer)
          - ood_class_acc    : classifier accuracy on f(x_noisy) vs y_true
          - ood_clean_acc    : classifier accuracy on x_clean vs y_true (sanity check)
    """
    classifier.eval()
    z = torch.randn_like(x_clean)
    x_noisy = (x_clean + noise_sigma * z).clamp(-1, 1)

    fx_noisy = ign_model(x_noisy)

    input_l1 = F.l1_loss(x_noisy, x_clean).item()
    proj_l1 = F.l1_loss(fx_noisy, x_clean).item()

    proj_logits = classifier(fx_noisy)
    proj_acc = (proj_logits.argmax(-1) == y_true).float().mean().item()

    clean_logits = classifier(x_clean)
    clean_acc = (clean_logits.argmax(-1) == y_true).float().mean().item()

    return {
        'ood_input_l1': float(input_l1),
        'ood_proj_l1': float(proj_l1),
        'ood_improvement': float(input_l1 - proj_l1),
        'ood_class_acc': float(proj_acc),
        'ood_clean_acc': float(clean_acc),
    }


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
