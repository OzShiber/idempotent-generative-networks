"""Latent-prior sampling diagnostic ("option 1" of the generation plan).

Question this answers: is the trained encoder g's latent space already
GENERATIVE — i.e., if we sample latents from a simple Gaussian fitted to the
latents of real images g(x) and decode with g^-1, do we get faces?

Why it matters: the flow-matching head produced noise despite a low flow loss,
and there are two candidate culprits:
  (a) the SAMPLER was too weak (a rank-256 linear correction on a 12288-dim
      latent cannot synthesize face structure), or
  (b) g's latent GEOMETRY doesn't support sampling at all (the set of
      face-latents is a thin curved sheet that no simple Gaussian covers).
This script needs NO training and separates the two:

  - samples look face-like (even blurry)  -> the latent space is usable;
    invest in a stronger sampler (UNet velocity field / compressed-latent
    flow — options 2/3).
  - samples are junk at EVERY temperature -> g's geometry is the bottleneck;
    fix g first (NLL term, compressor, input-dependent A — options 4/5).

Priors fitted to N latents of real training images (Xf = g(x) flattened):
  diag     u = mean + T * std ⊙ eps                 per-element diagonal fit
  pca{K}   u = mean + T * (top-K PCA + diagonal residual) sample
Each prior is decoded two ways — g^-1(u) and g^-1(A u) (the f-path) — at
temperatures 1.0 / 0.7 / 0.5. Reduced temperature (Glow-style) trades
diversity for fidelity and is often the difference between junk and faces.
The SAME eps is reused across temperatures, so within a grid each column is
one sample rendered at three temperatures (rows: T=1.0, 0.7, 0.5).

Also saved, all under <exp_dir>/prior_samples/:
  real_vs_fx.png       sanity anchor — top row real x, bottom row f(x)
  fz_pixelnoise.png    the current failure reference — f(z) for pixel noise
                       (top row: uniform z as in training valid grids;
                        bottom row: Gaussian z)
  mean_face.png        decoded mean latent [g^-1(mean), g^-1(A mean)] — the
                       "average face" of the latent space
  interp_latent.png    latent-space interpolation between real image pairs
                       (rows = pairs, columns = alpha 0..1) — smoothness check
  prior_stats.txt      latent norms + whitened distances + PCA explained
                       variance. Key number: how far pixel-noise latents g(z)
                       sit from the fitted data-latent Gaussian (whitened
                       mean-square ≈ 1.0 means in-distribution). Also, if the
                       top-K PCA explains most of the variance, the latent is
                       effectively low-dimensional and a compressed-latent
                       flow (option 3) has an easy target.

Run via:  python train_ign.py --mode prior_sample ...  (same contract as test
mode: the architecture flags MUST match the trained checkpoint).
"""

import os
import torch
from torchvision.utils import make_grid, save_image
from data import get_denormalize_fn


TEMPERATURES = (1.0, 0.7, 0.5)
N_SHOW = 8          # samples per grid row
DECODE_BS = 16      # g.inverse batch size (a 30L w256 net is heavy)


@torch.no_grad()
def _decode(model, U, device, apply_A=False):
    """Decode latents U (N, C, H, W) to images via g^-1, optionally projecting
    through the idempotent operator A first (the f-path)."""
    outs = []
    for i in range(0, U.shape[0], DECODE_BS):
        u = U[i:i + DECODE_BS].to(device)
        if apply_A:
            u = model.A(u)
        outs.append(model.g.inverse(u).cpu())
    return torch.cat(outs)


def _save_grid(imgs, path, denorm, nrow):
    save_image(make_grid(denorm(imgs).clamp(0, 1), nrow=nrow, padding=2), path)
    print(f"[prior] saved {path}")


@torch.no_grad()
def run_prior_sampling(model, train_loader, conf):
    device = conf.device
    model.eval()
    denorm = get_denormalize_fn(conf.dataset)
    out_dir = os.path.join(conf.exp_dir, "prior_samples")
    os.makedirs(out_dir, exist_ok=True)

    im_shape = tuple(conf.im_shape)
    D = im_shape[0] * im_shape[1] * im_shape[2]
    n_fit = int(getattr(conf, 'prior_n_fit', 4096))
    pca_rank = int(getattr(conf, 'prior_pca_rank', 256))

    # ---- 1. Collect latents of real images --------------------------------
    print(f"[prior] encoding {n_fit} real images through g ...")
    lat, real = [], []
    n = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        lat.append(model.g(xb).cpu())
        if sum(r.shape[0] for r in real) < 2 * N_SHOW:
            real.append(xb.cpu())
        n += xb.shape[0]
        if n >= n_fit:
            break
    X = torch.cat(lat)[:n_fit]                       # (N, C, H, W) latents
    real = torch.cat(real)[:2 * N_SHOW]              # a few real images
    N = X.shape[0]
    Xf = X.reshape(N, -1).to(device)                 # (N, D)
    mean = Xf.mean(0)                                # (D,)
    std = Xf.std(0).clamp_min(1e-6)                  # (D,)

    # ---- 2. PCA (low-rank + diagonal-residual Gaussian) -------------------
    Xc = Xf - mean
    q = min(pca_rank, N - 1, D)
    print(f"[prior] fitting PCA rank {q} on ({N}, {D}) latents ...")
    _, S, V = torch.pca_lowrank(Xc, q=q, center=False)   # V: (D, q)
    coeff = Xc @ V                                       # (N, q)
    coeff_std = coeff.std(0).clamp_min(1e-6)             # (q,)
    resid_std = (Xc - coeff @ V.t()).std(0)              # (D,)
    total_var = Xc.var(0).sum().item()
    explained = coeff.var(0).sum().item() / max(total_var, 1e-12)

    # ---- 3. Prior sample grids (rows = temperatures, same eps per column) --
    eps_diag = torch.randn(N_SHOW, D, device=device)
    eps_lr = torch.randn(N_SHOW, q, device=device)
    eps_res = torch.randn(N_SHOW, D, device=device)

    def diag_latents(T):
        return (mean + T * std * eps_diag).reshape(N_SHOW, *im_shape)

    def pca_latents(T):
        u = mean + T * ((eps_lr * coeff_std) @ V.t() + eps_res * resid_std)
        return u.reshape(N_SHOW, *im_shape)

    for name, latents_fn in (("diag", diag_latents), (f"pca{q}", pca_latents)):
        for apply_A, suffix in ((False, "g"), (True, "Ag")):
            rows = [_decode(model, latents_fn(T), device, apply_A=apply_A)
                    for T in TEMPERATURES]
            _save_grid(torch.cat(rows),
                       os.path.join(out_dir, f"prior_{name}_{suffix}.png"),
                       denorm, nrow=N_SHOW)

    # ---- 4. Baselines ------------------------------------------------------
    # Sanity anchor: real images vs their f(x) projections (restoration path).
    x_real = real[:N_SHOW].to(device)
    fx = model(x_real).cpu()
    _save_grid(torch.cat([x_real.cpu(), fx]),
               os.path.join(out_dir, "real_vs_fx.png"), denorm, nrow=N_SHOW)

    # Failure reference: f(z) on pixel noise — what generation currently does.
    z_uniform = torch.rand(N_SHOW, *im_shape, device=device) * 2. - 1.
    z_gauss = torch.randn(N_SHOW, *im_shape, device=device)
    _save_grid(torch.cat([model(z_uniform).cpu(), model(z_gauss).cpu()]),
               os.path.join(out_dir, "fz_pixelnoise.png"), denorm, nrow=N_SHOW)

    # The "average face": decoded mean latent, plain and A-projected.
    u_mean = mean.reshape(1, *im_shape)
    _save_grid(torch.cat([_decode(model, u_mean, device, apply_A=False),
                          _decode(model, u_mean, device, apply_A=True)]),
               os.path.join(out_dir, "mean_face.png"), denorm, nrow=2)

    # Latent interpolation between real pairs — smoothness of g's geometry.
    # If midpoints decode to plausible faces, the latent set is convex-ish
    # around data and Gaussian sampling has a chance; if midpoints shatter,
    # the manifold is thin/curved and needs a learned sampler regardless.
    n_pairs = 4
    g_a = model.g(real[:n_pairs].to(device))
    g_b = model.g(real[n_pairs:2 * n_pairs].to(device))
    alphas = torch.linspace(0, 1, N_SHOW, device=device)
    interp = torch.cat([(1 - a) * g_a + a * g_b for a in alphas], dim=0)
    # Reorder so each ROW is one pair swept over alpha (currently grouped by alpha).
    interp = interp.reshape(N_SHOW, n_pairs, *im_shape).transpose(0, 1).reshape(-1, *im_shape)
    _save_grid(_decode(model, interp, device),
               os.path.join(out_dir, "interp_latent.png"), denorm, nrow=N_SHOW)

    # ---- 5. Latent statistics ----------------------------------------------
    g_zu = model.g(torch.rand(64, *im_shape, device=device) * 2. - 1.).reshape(64, -1)
    g_zg = model.g(torch.randn(64, *im_shape, device=device)).reshape(64, -1)

    def norm_stats(M):
        nn = M.norm(dim=1)
        return nn.mean().item(), nn.std().item()

    def whitened(M):
        # Mean squared per-element z-score under the fitted diagonal Gaussian.
        # ~1.0 = statistically indistinguishable from real-image latents.
        return (((M - mean) / std) ** 2).mean().item()

    lines = [
        f"n_fit={N}  D={D}  pca_rank={q}  im_shape={im_shape}",
        f"PCA explained variance (top {q}): {explained:.4f}",
        f"top-10 singular values: {[round(s, 2) for s in S[:10].tolist()]}",
        "",
        "latent L2 norms (mean +- std over batch):",
        "  g(real x)        : {:.2f} +- {:.2f}".format(*norm_stats(Xf)),
        "  g(z) uniform pix : {:.2f} +- {:.2f}".format(*norm_stats(g_zu)),
        "  g(z) gauss pix   : {:.2f} +- {:.2f}".format(*norm_stats(g_zg)),
        "",
        "whitened mean-square under fitted diag Gaussian (1.0 = in-dist):",
        f"  g(real x)        : {whitened(Xf):.3f}",
        f"  g(z) uniform pix : {whitened(g_zu):.3f}",
        f"  g(z) gauss pix   : {whitened(g_zg):.3f}",
        "",
        "grids: rows are temperatures T=1.0 / 0.7 / 0.5 (same eps per column).",
        "  prior_diag_g.png    diag-Gaussian sample, decoded g^-1(u)",
        "  prior_diag_Ag.png   same latents through the f-path g^-1(A u)",
        f"  prior_pca{q}_g.png / _Ag.png   low-rank+residual Gaussian sample",
        "  real_vs_fx.png      row1 real x, row2 f(x)",
        "  fz_pixelnoise.png   row1 f(uniform z), row2 f(gauss z)",
        "  mean_face.png       [g^-1(mean), g^-1(A mean)]",
        "  interp_latent.png   rows = real pairs, cols = alpha 0..1",
    ]
    stats_path = os.path.join(out_dir, "prior_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[prior] " + "\n[prior] ".join(lines))
    print(f"[prior] stats written to {stats_path}")
    print(f"[prior] done — all outputs under {out_dir}")
