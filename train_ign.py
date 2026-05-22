from ast import Yield
import os
import argparse
import random
import torch
import numpy as np
import pickle
from utils import create_experiment_dirs_ign, make_ign_inputs
from lin_ign import LinearIGN
from data import get_data_loaders
from torchvision.utils import save_image
from torchvision.utils import make_grid, save_image
from data import mnist_denormalize as denorm
from data import mnist_normalize as norm



def main():
    parser = argparse.ArgumentParser(description="Train or Test a Linear Idempotent Generative Network (IGN)")
    
    # General training arguments
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Whether to train a new model or run test-time projections.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for Python, NumPy, and PyTorch (CPU + CUDA). "
                             "Single point of control for run-to-run reproducibility. "
                             "Two runs with the same --seed and identical configuration should "
                             "produce identical trajectories (modulo nondeterministic CUDA ops). "
                             "Use different seeds for multi-seed ablations.")
    parser.add_argument("--n_epochs", type=int, default=50000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size (for grid).")
    parser.add_argument("--log_freq", type=int, default=20, help="Log frequency (in steps).")
    parser.add_argument("--save_val_ckpt", action=argparse.BooleanOptionalAction, default=True, help="Save a checkpoint after validation.")
    parser.add_argument("--val_freq", type=int, default=1, help="Validation frequency (in epochs).")
    parser.add_argument("--ckpt", type=str, default='/home/assaf.sh/projects/Linearizer/results/ZZZ_mnist_20250921_151747_ign_mnist_BASE/ckpts/e903.pth', help="Checkpoint to load: 'latest' or a filename.")
    # parser.add_argument("--ckpt", type=str, default='/home/assaf.sh/projects/Linearizer/results/mnist_20250922_013040_ign_mnist/ckpts/e4929.pth', help="Checkpoint to load: 'latest' or a filename.")
    # parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default='/home/assaf.sh/projects/Linearizer/results/mnist_20250922_013245_ign_mnist/ckpts/e4929.pth', help="Checkpoint to load: 'latest' or a filename.")
    
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay (L2 penalty).")
    
    # Experiment & folder settings
    parser.add_argument("--results_dir", type=str, default="./results", help="Base directory to save experiment results.")
    parser.add_argument("--exp_name", type=str, default="ign_mnist", help="Experiment folder name.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Use WandB for logging.")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use.")
    parser.add_argument("--orig_im_size", type=int, default=28, help="Original image size.")
    parser.add_argument("--target_im_size", type=int, default=32, help="Target image size for the model.")
    parser.add_argument("--im_shape", type=int, nargs=3, default=[1, 32, 32], help="Image shape as C H W.")

    # Model-specific arguments
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in invertible network g.")
    parser.add_argument("--hidden_chans", type=int, default=128,
                        help="Width of each CNN coupling block — controls the channel pyramid: "
                             "the block goes 2 → h/16 → h/4 → h → h*4 → h → h/4 → h/16 → 2. "
                             "Default 128 reproduces the original (peak 512). 256 = ~2x wider, ~4x params per block. "
                             "Must be divisible by 16. Doubling triggers a noticeable GPU-memory increase; "
                             "expect to drop --batch_size from 128 to ~64 at 256 width.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads in g.")
    parser.add_argument("--p_sz", type=int, default=4, help="Patch size for invertible transformer g.")
    parser.add_argument("--binarizer", type=str, choices=['rotation', 'ste', 'gumbel'], default='rotation',
                        help="Gradient estimator for the binary diagonal A. 'rotation' = current default (RotationTrickEstimator). 'ste' = plain straight-through. 'gumbel' = Gumbel-sigmoid + STE.")
    parser.add_argument("--gumbel_tau", type=float, default=0.5, help="Temperature for the gumbel binarizer (ignored for ste/rotation).")
    parser.add_argument("--a_operator", type=str, default="diagonal",
                        choices=["diagonal", "projection"],
                        help="Form of the idempotent operator A. "
                             "'diagonal' (default): A = diag(b), b in {0,1}^D — projection onto a coordinate-aligned K-dim subspace. "
                             "'projection': A = Q L Q^T, where Q is a learned orthogonal "
                             "(parametrized as a product of Householder reflections) and "
                             "L is the same binary diagonal. Lets the model learn the orientation of "
                             "its projection subspace, not just which axes to keep. Idempotent by "
                             "construction either way. 'projection' adds n_householders * D parameters for Q.")
    parser.add_argument("--n_householders", type=int, default=64,
                        help="Number of Householder reflections parametrizing Q for "
                             "--a_operator projection. K=D would span all of SO(D); "
                             "K << D is sufficient if Q only needs to align a few principal "
                             "directions. 64 is a sensible starting value; ignored when "
                             "--a_operator diagonal.")

    # Loss weights — defaults to 0 disable each auxiliary loss; the code paths still run for instrumentation.
    parser.add_argument("--lambda_rec", type=float, default=1., help="Weight for the reconstruction loss.")
    parser.add_argument("--lambda_sparse", type=float, default=0.0, help="Weight for the A sparsity loss. 0 = disabled (loss still computed and logged).")
    parser.add_argument("--lambda_tight", type=float, default=0.0, help="Weight for the A tightness loss. 0 = disabled (loss still computed and logged).")
    parser.add_argument("--lambda_denoise", type=float, default=0.0, help="Weight for the noisy-reconstruction (denoising) loss. 0 = disabled (loss still computed and logged).")
    parser.add_argument("--noise_sigma", type=float, default=0.3, help="Noise std added to x for the denoising loss.")
    parser.add_argument("--lambda_feat", type=float, default=0.0,
                        help="Weight for the latent-space feature-matching loss L1(g(f(x)), g(x).detach()). "
                             "Encourages f's reconstructions to match the original at a coarser, structural "
                             "level than per-pixel L1. The detach on g(x) prevents g from saturating to "
                             "trivially satisfy the loss (a previously observed degenerate failure). "
                             "0 = disabled (skips the extra g forward pass; loss logged as 0). "
                             "Try 0.1 to enable; 0.05–0.5 is the sensible range. Adds ~one g forward "
                             "per training step (10–20%% slowdown depending on n_layers).")
    parser.add_argument("--lambda_classifier", type=float, default=0.0,
                        help="Weight for the pretrained-classifier perceptual loss "
                             "L1(D(f(x))_features, D(x)_features.detach()). Uses the FROZEN MNIST classifier "
                             "loaded via --eval_classifier_path as a feature extractor. Standard perceptual "
                             "loss (Johnson 2016) — pushes f's outputs to match real samples in a "
                             "semantically meaningful feature space, addressing the structural blur from L1. "
                             "Requires --eval_classifier_path to be set and the classifier file to exist. "
                             "0 = disabled. Try 0.1 to enable; 0.05–0.5 is the sensible range. Adds ~one "
                             "small classifier forward per training step (~5%% slowdown).")

    # Quantitative evaluation — classifier-based generation metrics + OOD projection.
    # Disabled unless --eval_classifier_path is set. Requires a one-time training
    # of the classifier via `python eval_metrics.py train --out <path>`.
    parser.add_argument("--eval_classifier_path", type=str, default="",
                        help="Path to a trained MNIST classifier (.pth) for quantitative metrics. "
                             "Train it once: `python eval_metrics.py train --out mnist_classifier.pth`. "
                             "Empty (default) = no quantitative metrics, only the visualization grid. "
                             "When set, every validation epoch logs gen_{entropy,confidence,coverage} on "
                             "a fixed batch of f(z), and ood_{input_l1,proj_l1,improvement,class_acc} on "
                             "a fixed batch of clean test images corrupted with --noise_sigma noise.")
    parser.add_argument("--eval_n_samples", type=int, default=256,
                        help="Number of samples used for quantitative eval (entropy / coverage / OOD). "
                             "Larger = more reliable but slightly more expensive at validation time.")

    conf = parser.parse_args()
    print(conf)

    # Seed all RNGs before constructing anything. Done as early as possible so
    # model init, data shuffling, and validation-noise sampling are all
    # deterministic given the same --seed. Two runs with identical config
    # and identical --seed should reproduce each other modulo nondeterministic
    # CUDA ops (matmul/conv kernels, atomic adds on GPU).
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    print(f"[seed] all RNGs set to {conf.seed}")

    if conf.wandb:
        try:
            import wandb
            wandb.init(config=conf, project="Linearizer_IGN", name=conf.exp_name)
        except ImportError:
            print("WandB not installed. Disabling WandB.")
            conf.wandb = False
            
    # --- Device handling ---
    conf.device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    
    # --- Experiment folder handling ---
    exp_dir, ckpt_dir, grid_dir = create_experiment_dirs_ign(conf.results_dir, conf.dataset, conf.exp_name)
    conf.exp_dir = exp_dir
    conf.ckpt_dir = ckpt_dir
    conf.grid_dir = grid_dir
    print(f"Experiment directory: {exp_dir}")

    # Save config
    with open(os.path.join(exp_dir, "conf.pkl"), "wb") as f:
        pickle.dump(conf, f)
    
    # --- Initialize the model ---
    model = LinearIGN(conf).to(conf.device)
    
    # Load checkpoint if requested
    if conf.ckpt:
        model.load_checkpoint(conf.ckpt if conf.ckpt.lower() != "latest" else None)

    # === Mode handling ===
    if conf.mode == "train":
        # --- Data loaders ---
        train_loader, _ = get_data_loaders(
            conf.dataset,
            conf.batch_size,
            conf.val_batch_size,
            conf.orig_im_size,
            conf.target_im_size)

        # --- Start training ---
        model.train_model(train_loader, conf.n_epochs)

        # Save final model
        final_ckpt_path = os.path.join(conf.ckpt_dir, "final_model.pth")
        torch.save(model.state_dict(), final_ckpt_path)
        print(f"Training complete. Final model saved to '{final_ckpt_path}'.")

    elif conf.mode == "test":

        for ind in range(200):
            # same loader as training
            train_loader, _ = get_data_loaders(
                conf.dataset,
                conf.batch_size,
                conf.val_batch_size,
                conf.orig_im_size,
                conf.target_im_size
            )

            # Build ALL inputs in [0,1] space
            inputs = make_ign_inputs(train_loader, size=conf.target_im_size)

            # Batch once
            names, tensors = zip(*inputs.items())
            X_raw = torch.cat([t for t in tensors], dim=0)          # [N,C,H,W], all in [0,1]
            X = (X_raw.clone()).to(conf.device)             # normalize entire batch for the model
            # add 1024 real samples from loader
            X = X.clamp(-1, 1)

            model.eval()
            with torch.no_grad():
                extra = []
                for xb, _ in train_loader:
                    extra.append(xb.to(conf.device))
                    if sum(len(e) for e in extra) >= 1024:
                        break
                X_extra = torch.cat(extra, dim=0)[:1024]

                # concat real + crafted
                X_in = torch.cat([X_extra, X], dim=0)

                # run once
                Y_in = model(X_in)

                # keep only crafted outputs
                Y = Y_in[-X.size(0):]

                Y_disp = (Y).cpu().clamp(0,1).cpu()     
                Y = Y.view(Y.shape[0], -1)
                
                Y = (Y-Y.min(1, keepdim=True)[0]) / (Y.max(1, keepdim=True)[0]-Y.min(1, keepdim=True)[0])     
                
                Y = Y.view_as(X)       

            # 2-row grid: top inputs (raw), bottom outputs (denormed)
            from torchvision.utils import make_grid, save_image
            n = X.shape[0]
            grid_in  = make_grid(X_raw.clamp(0,1), nrow=n, padding=2)   # inputs in [0,1]
            grid_out = make_grid(Y_disp,            nrow=n, padding=2)  # outputs denormed to [0,1]
            grid = torch.cat([grid_in, grid_out], dim=1)

            out_dir = os.path.join(exp_dir, "free_inputs")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"ign_res_{ind}.png")
            save_image(grid, save_path)

            with open(os.path.join(out_dir, "order.txt"), "w") as f:
                f.write("\n".join([f"{i:02d}: {name}" for i, name in enumerate(names)]))

            print(f"[test] saved grid to {save_path}")

if __name__ == "__main__":
    main()
