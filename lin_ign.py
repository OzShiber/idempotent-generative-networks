import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import InvTransformerNet, IdempotentDiagonalOperator, test_model_properties, BasicLinearizer, InvCNNNet
from torchvision.utils import make_grid
from utils import imwrite, find_latest_checkpoint
from data import mnist_denormalize as denorm
import wandb
from song__unet import creat_song_unet
import csv
from inv_Unet import InvUnetV2


class LinearIGN(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        if not isinstance(self.conf.im_shape, tuple):
            self.conf.im_shape = tuple(self.conf.im_shape)
        
        # Invertible network 'g'
        # self.g = InvTransformerNet(conf.n_heads, conf.n_layers, conf.p_sz, self.conf.im_shape[-1], rgb=(self.conf.im_shape[0] == 3))
        self.g = InvCNNNet(conf.n_layers, conf.im_shape[-1])
        #self.g = InvUnetV2(num_layers=1, in_channels=1, im_sz=32, unet_creator=creat_song_unet)
        
        # Idempotent linear operator 'A'
        input_dim = self.conf.im_shape[0] * self.conf.im_shape[1] * self.conf.im_shape[2]
        self.A = IdempotentDiagonalOperator(
            input_dim,
            binarizer=getattr(self.conf, 'binarizer', 'rotation'),
            gumbel_tau=getattr(self.conf, 'gumbel_tau', 0.5),
        )
        
        # The complete Linearizer model f(x)
        self.model = BasicLinearizer(self.g, self.A)

        # Attributes for compatibility with test_model_properties
        self.gx = self.gy = self.g 
        self.rgb = (self.conf.im_shape[0] == 3)

        # Fixed noise for consistent validation visuals
        self.register_buffer("valid_z", torch.rand(conf.val_batch_size, *conf.im_shape)*2.-1.)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x, ret_intermid=False):
        return self.model(x, ret_intermid)

    def train_step(self, x, z):
        # ret_intermid=True returns (y_pred, g_x, g_y_pred); g_x is g(x) already.
        fx, gx, _ = self(x, ret_intermid=True)

        # --- L_rec: per-pixel L1 reconstruction. Median-seeking, so partly
        # responsible for the model's blur — but switching to L2 is reliably worse.
        loss_rec = torch.nn.functional.l1_loss(fx, x)

        # Testing L2 Loss
        #loss_rec = torch.nn.functional.mse_loss(fx, x)

        # --- L_sparse (flat): constant downward pressure on A.diag proportional
        # to the fraction of active dims. Under STE this is what keeps A from
        # collapsing toward identity; under rotation trick the implicit
        # norm-scaling already provides this pressure, so a flat formulation
        # is redundant but not harmful (set lambda_sparse=0 to disable).
        loss_sparse = self.A.diag.mean()

        # --- L_tight: kept for instrumentation; lambda_tight=0 by default.
        # Empirically this loss grows during training (opposite of intent) and
        # has been shown to harm generation quality even at small weights.
        zero = self.g.inverse(torch.zeros_like(x[:1]))
        loss_tight = (gx.pow(2).mean((1, 2, 3)) - (x - zero).pow(2).mean((1, 2, 3))).abs().mean()

        # --- L_denoise: f(x + sigma*z) should match x. Direct projection signal
        # for f when fed non-real inputs. Empirically not strong enough on its own
        # to drive useful generation; lambda_denoise=0 by default.
        x_noisy = (x + self.conf.noise_sigma * z).clamp(-1, 1)
        fx_noisy = self(x_noisy)
        loss_denoise = torch.nn.functional.l1_loss(fx_noisy, x)

        # --- L_feat (Option 1a, latent feature matching, detach variant):
        # match g(x) and g(f(x)) in g's latent space, in addition to matching
        # x and f(x) in pixel space.
        #
        # The motivation: pixel L1 is a single global average that can't distinguish
        # "off by a smear of low-frequency error" from "off by sharp high-frequency
        # errors with the same pixel-sum" — both look like the same number. g is a
        # multi-scale invertible transform (each InvCNN layer halves spatial size
        # and packs structure into channels), so L1(g(x), g(f(x))) penalises the
        # difference at a coarser, more structural level.
        #
        # CRITICAL: gx is detached. Gradient flows ONLY through the g(f(x)) side
        # (and through f(x) back into f's parameters). Without the detach, g gets
        # gradient from both sides of the loss, and the optimizer's easiest path
        # is to saturate g's output range so that g(x) and g(f(x)) trivially match
        # regardless of how well f reconstructs. This was observed empirically at
        # lambda_feat=0.1 (run mnist_20260502_214746): loss_feat collapsed to ~0
        # within 11 epochs, rec regressed 0.17→0.20 from ep26 to ep50, samples
        # mode-collapsed to 6-shapes. Detaching gx breaks that escape — g's
        # representation is a fixed target for this loss, so f has to actually
        # produce reconstructions that match in latent space.
        #
        # Cost: one extra forward through g per training step.
        # Disabled by default with lambda_feat=0.
        if self.conf.lambda_feat > 0:
            g_fx = self.g(fx)
            loss_feat = torch.nn.functional.l1_loss(g_fx, gx.detach())
        else:
            # Skip the extra forward when disabled — keep the metric for logging
            # so its column in metrics.csv stays consistent.
            loss_feat = torch.zeros((), device=x.device)

        total_loss = (self.conf.lambda_rec * loss_rec +
                      self.conf.lambda_sparse * loss_sparse +
                      self.conf.lambda_tight * loss_tight +
                      self.conf.lambda_denoise * loss_denoise +
                      self.conf.lambda_feat * loss_feat)

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
        self.opt.step()


        return (total_loss.item(), loss_rec.item(), loss_sparse.item(),
                loss_tight.item(), loss_denoise.item(), loss_feat.item())

    def train_model(self, train_loader, n_epochs):
        self.opt = optim.Adam(self.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs)
        
        device = self.device()
        global_counter = 0
        
        print("--- Starting Training for LinearIGN ---")
        for epoch in range(n_epochs):
                       
            running_loss, running_rec, running_sparse, running_tight, running_denoise, running_feat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            counter = 0
            num_batches = len(train_loader)

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                z = torch.randn_like(x)

                loss, loss_rec, loss_sparse, loss_tight, loss_denoise, loss_feat = self.train_step(x, z)

                B = x.shape[0]
                counter += B
                global_counter += 1
                running_loss += loss * B
                running_rec += loss_rec * B
                running_sparse += loss_sparse * B
                running_tight += loss_tight * B
                running_denoise += loss_denoise * B
                running_feat += loss_feat * B

                if (global_counter) % self.conf.log_freq == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    avg_loss = running_loss / counter
                    avg_rec = running_rec / counter
                    avg_sparse = running_sparse / counter
                    avg_tight = running_tight / counter
                    avg_denoise = running_denoise / counter
                    avg_feat = running_feat / counter
                    # Diagnostics on A's learned diagonal
                    A_active = self.A.diag.sum().item()        # number of dims with binary value 1
                    A_total = self.A.diag.numel()
                    A_probs_mean = self.A.probs.mean().item()  # soft fraction (sigmoid before rounding)
                    print(f"[Train] Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{num_batches}] "
                          f"LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Rec: {avg_rec:.4f} | Sparse: {avg_sparse:.4f} | tight: {avg_tight:.4f} | denoise: {avg_denoise:.4f} | feat: {avg_feat:.4f} | A_active: {A_active:.0f}/{A_total} | A_probs_mean: {A_probs_mean:.3f}")
                    metrics = {
                    'step': global_counter, 'epoch': epoch+1, 'lr': current_lr,
                    'loss': avg_loss, 'rec': avg_rec, 'sparse': avg_sparse, 'tight': avg_tight, 'denoise': avg_denoise, 'feat': avg_feat,
                    'A_active': A_active, 'A_probs_mean': A_probs_mean
                    }
                    self.log_local_metrics(metrics)

                    if self.conf.wandb:
                        wandb.log({
                            'LR': current_lr, 'loss': avg_loss,
                            'loss_rec': avg_rec,
                            'loss_sparse': avg_sparse,
                            'loss tight': avg_tight,
                            'loss_denoise': avg_denoise,
                            'loss_feat': avg_feat,
                            'A_active': A_active,
                            'A_probs_mean': A_probs_mean
                        }, step=global_counter)
                    counter = 0
                    running_loss, running_rec, running_sparse, running_tight, running_denoise, running_feat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            if (epoch + 1) % self.conf.val_freq == 0:
                self.valid(epoch)

            self.sched.step()
    
    @torch.no_grad()
    def valid(self, epoch):
        self.eval()
        print("\n--- Running Validation ---")
        # test_model_properties(self) # Can be slow, optional

        # Generate samples from fixed noise
        gen_samples = self(self.valid_z)

        # Create a grid of z vs f(z)
        combined_grid = denorm(gen_samples).clip(0, 1)
        grid = make_grid(combined_grid, nrow=self.conf.val_batch_size)
        
        path = os.path.join(self.conf.grid_dir, f"samples_e{epoch+1}.png")
        imwrite(grid, path)
        print(f"Saved validation samples to {path}")

        if self.conf.wandb:
            wandb.log({"Samples (z vs f(z))": wandb.Image(path), "epoch": epoch})

        if self.conf.save_val_ckpt:
            self.save_checkpoint(f"e{epoch+1}.pth")

        self.train()
        print("--- Validation Complete ---\n")


    def save_checkpoint(self, filename):
        ckpt_path = os.path.join(self.conf.ckpt_dir, filename)
        torch.save(self.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    
    def load_checkpoint(self, ckpt_file=None):
        if ckpt_file is None:
            ckpt_file = find_latest_checkpoint(self.conf.ckpt_dir)
            if ckpt_file is None:
                print("No checkpoint found.")
                return
        try:
            print(f'Will now load ckpt from file {ckpt_file}')
            ckpt = torch.load(ckpt_file, map_location=self.device())
            self.load_state_dict(ckpt)
            print(f"Loaded checkpoint from {ckpt_file}")
        except FileNotFoundError:
            print(f"Checkpoint file not found: {ckpt_file}")
    
    def log_local_metrics(self, metrics):
    
        log_path = os.path.join(self.conf.exp_dir, "metrics.csv")
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
