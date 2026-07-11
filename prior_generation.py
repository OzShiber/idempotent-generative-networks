"""Latent-prior GENERATION (step 1 of the generation fix).

The prior_sample diagnostic (June run) showed:
  - pixel-noise latents g(z) land 5-15x outside the real-latent distribution
    (whitened distance 5.2 / 15.3 vs 1.0) -> decoding them gives noise;
  - latents sampled from a Gaussian FITTED to real g(x) decode to faces;
  - 256 of 12,288 latent dims explain 84% of the variance.

So this mode turns that finding into the model's actual generator:

  1. fit a low-rank Gaussian prior to g(x) over the training set
       u = mean + T * ( (eps_q * coeff_std) @ V^T  +  eps_D * resid_std )
     (PCA top-q directions + diagonal residual; q = --prior_pca_rank)
  2. sample it at several temperatures and decode:
       g^-1(A u)   -- the f-path: samples projected onto the manifold (main)
       g^-1(u)     -- raw decode (comparison)
  3. score it with FID (ImageNet-ResNet18 features, same extractor as the
     training-time eval) against three references:
       fid_prior_Ag / fid_prior_g : the new generator
       fid_fz_pixel               : the OLD generation path f(z) -- baseline
       fid_real_floor             : FID(real train, real test) -- the floor
                                    any generator is bounded below by

Since g is invertible, "fit a Gaussian to g(x), sample, decode" is exactly
how a normalizing flow generates — the model was a generator all along; only
the sampling location was wrong. This is NOT the rejected "structured noise"
idea: nothing image-like is fed in; the prior is part of the model.

Outputs (under <exp_dir>/prior_gen/):
  prior_gen_T{t}.png      8x8 sample sheets at T in {1.0, 0.85, 0.7} (f-path)
  prior_gen_T1.0_g.png    raw-decode sheet at T=1.0 (comparison)
  latent_prior.pth        the fitted prior (mean, V, coeff_std, resid_std)
  prior_gen_stats.txt     fit meta + the FID table

Run via train_ign.py --mode prior_gen (architecture flags MUST match the
trained checkpoint, same contract as test / prior_sample modes).
"""

import os
import torch
from torchvision.utils import make_grid, save_image
from data import get_denormalize_fn


TEMPERATURES = (1.0, 0.85, 0.7)
N_SHEET = 64        # samples per sheet (8x8)
DECODE_BS = 32      # g.inverse batch size


@torch.no_grad()
def fit_latent_prior(model, train_loader, conf, keep_real=0):
    """Encode training images through g and fit the low-rank+diagonal Gaussian.

    Returns (prior_dict, real_kept) where real_kept is the first `keep_real`
    training images (CPU) for the FID floor reference. prior tensors live on
    conf.device; the saved copy is moved to CPU by the caller.
    """
    device = conf.device
    n_fit = int(getattr(conf, 'prior_gen_n_fit', 8192))
    q = int(getattr(conf, 'prior_pca_rank', 256))

    print(f"[prior_gen] encoding {n_fit} real images through g ...")
    lat, real = [], []
    n = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        lat.append(model.g(xb).cpu())
        if keep_real and sum(r.shape[0] for r in real) < keep_real:
            real.append(xb.cpu())
        n += xb.shape[0]
        if n >= n_fit:
            break
    X = torch.cat(lat)[:n_fit]
    real_kept = torch.cat(real)[:keep_real] if real else None
    N = X.shape[0]
    Xf = X.reshape(N, -1).to(device)

    mean = Xf.mean(0)
    Xc = Xf - mean
    q = min(q, N - 1, Xf.shape[1])
    print(f"[prior_gen] fitting PCA rank {q} on ({N}, {Xf.shape[1]}) latents ...")
    _, S, V = torch.pca_lowrank(Xc, q=q, center=False)          # V: (D, q)
    coeff = Xc @ V                                              # (N, q)
    coeff_std = coeff.std(0).clamp_min(1e-6)
    resid_std = (Xc - coeff @ V.t()).std(0)
    explained = (coeff.var(0).sum() / Xc.var(0).sum().clamp_min(1e-12)).item()

    prior = {
        'mean': mean, 'V': V, 'coeff_std': coeff_std, 'resid_std': resid_std,
        'n_fit': N, 'q': q, 'explained': explained,
        'im_shape': tuple(conf.im_shape),
    }
    return prior, real_kept


@torch.no_grad()
def sample_latent_prior(prior, n, temp=1.0, device=None):
    """Draw n latents from the fitted prior at temperature `temp`."""
    device = device or prior['mean'].device
    mean, V = prior['mean'].to(device), prior['V'].to(device)
    cs, rs = prior['coeff_std'].to(device), prior['resid_std'].to(device)
    eps_q = torch.randn(n, V.shape[1], device=device)
    eps_d = torch.randn(n, V.shape[0], device=device)
    u = mean + temp * ((eps_q * cs) @ V.t() + eps_d * rs)
    return u.reshape(n, *prior['im_shape'])


@torch.no_grad()
def _decode(model, U, device, apply_A=True):
    outs = []
    for i in range(0, U.shape[0], DECODE_BS):
        u = U[i:i + DECODE_BS].to(device)
        if apply_A:
            u = model.A(u)
        outs.append(model.g.inverse(u).cpu())
    return torch.cat(outs)


@torch.no_grad()
def _f_of(model, X, device):
    """f(X) in decode-sized batches (full forward: g -> A -> g^-1)."""
    outs = []
    for i in range(0, X.shape[0], DECODE_BS):
        outs.append(model(X[i:i + DECODE_BS].to(device)).cpu())
    return torch.cat(outs)


@torch.no_grad()
def run_prior_generation(model, train_loader, test_loader, conf):
    device = conf.device
    model.eval()
    denorm = get_denormalize_fn(conf.dataset)
    out_dir = os.path.join(conf.exp_dir, "prior_gen")
    os.makedirs(out_dir, exist_ok=True)
    n_fid = int(getattr(conf, 'prior_gen_n_fid', 1024))

    # ---- 1. fit the prior (keep n_fid real train images for the FID floor) --
    prior, real_train = fit_latent_prior(model, train_loader, conf, keep_real=n_fid)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in prior.items()},
               os.path.join(out_dir, "latent_prior.pth"))
    print(f"[prior_gen] prior saved; explained variance (top {prior['q']}): "
          f"{prior['explained']:.4f}")

    # ---- 2. sample sheets --------------------------------------------------
    for T in TEMPERATURES:
        u = sample_latent_prior(prior, N_SHEET, temp=T, device=device)
        sheet = _decode(model, u, device, apply_A=True)
        path = os.path.join(out_dir, f"prior_gen_T{T}.png")
        save_image(make_grid(denorm(sheet).clamp(0, 1), nrow=8, padding=2), path)
        print(f"[prior_gen] saved {path}")
    u = sample_latent_prior(prior, N_SHEET, temp=1.0, device=device)
    sheet = _decode(model, u, device, apply_A=False)
    path = os.path.join(out_dir, "prior_gen_T1.0_g.png")
    save_image(make_grid(denorm(sheet).clamp(0, 1), nrow=8, padding=2), path)
    print(f"[prior_gen] saved {path}")

    # ---- 3. FID table -------------------------------------------------------
    stats = [
        f"n_fit={prior['n_fit']}  q={prior['q']}  "
        f"explained={prior['explained']:.4f}  n_fid={n_fid}",
    ]
    try:
        from eval_metrics import ImageNetFeatureExtractor, _extract_features, compute_fid
        from data import get_normalize_stats
        mean_std = get_normalize_stats(conf.dataset)
        clf = ImageNetFeatureExtractor(input_mean=mean_std[0], input_std=mean_std[1]).to(device)

        # real references: test set (target) and a disjoint train set (floor)
        reals = []
        for xb, _ in test_loader:
            reals.append(xb)
            if sum(r.shape[0] for r in reals) >= n_fid:
                break
        real_test = torch.cat(reals)[:n_fid].to(device)
        feats_test = _extract_features(real_test, clf)

        def fid_of(imgs):
            return compute_fid(feats_test, _extract_features(imgs.to(device), clf))

        # the new generator, both decode paths (T=1.0)
        u = sample_latent_prior(prior, n_fid, temp=1.0, device=device)
        fid_Ag = fid_of(_decode(model, u, device, apply_A=True))
        fid_g = fid_of(_decode(model, u, device, apply_A=False))
        # the OLD generation path: f(uniform pixel noise) — the baseline
        z = torch.rand(n_fid, *conf.im_shape) * 2. - 1.
        fid_fz = fid_of(_f_of(model, z, device))
        # the floor: real train vs real test
        fid_floor = fid_of(real_train) if real_train is not None else float('nan')

        stats += [
            "",
            "FID vs real test images (lower = better; floor = real-vs-real):",
            f"  fid_prior_Ag (new generator, f-path) : {fid_Ag:10.2f}",
            f"  fid_prior_g  (new generator, raw)    : {fid_g:10.2f}",
            f"  fid_fz_pixel (OLD path, f(z))        : {fid_fz:10.2f}",
            f"  fid_real_floor (train vs test)       : {fid_floor:10.2f}",
        ]
    except Exception as e:  # keep the grids even if the extractor is unavailable
        stats += ["", f"FID SKIPPED: {type(e).__name__}: {e}"]
        print(f"[prior_gen] WARNING: FID skipped ({e})")

    stats_path = os.path.join(out_dir, "prior_gen_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats) + "\n")
    print("[prior_gen] " + "\n[prior_gen] ".join(stats))
    print(f"[prior_gen] done — outputs under {out_dir}")
