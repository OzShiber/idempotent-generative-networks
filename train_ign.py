from ast import Yield
import os
import argparse
import torch
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
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in invertible network g.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads in g.")
    parser.add_argument("--p_sz", type=int, default=4, help="Patch size for invertible transformer g.")
    
    # Loss weights
    parser.add_argument("--lambda_rec", type=float, default=1., help="Weight for the reconstruction loss.")
    parser.add_argument("--lambda_sparse", type=float, default=0.75, help="Weight for the A sparsity loss.")
    parser.add_argument("--lambda_tight", type=float, default=0.001, help="Weight for the A tightness loss.")

    conf = parser.parse_args()
    print(conf)
    
    if conf.wandb:
        try:
            import wandb
            wandb.init(config=conf, project="Linearizer_IGN")
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