import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import InvTransformerNet, IdempotentDiagonalOperator, test_model_properties, BasicLinearizer, InvCNNNet
from torchvision.utils import make_grid
from utils import imwrite, find_latest_checkpoint
from data import get_denormalize_fn
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
        # data_channels = number of image channels (1 for MNIST, 3 for CIFAR/etc.).
        # Threaded down to SpatialSplit, ActNorm, and CNNBlock's in_chans so the
        # invertible pipeline handles arbitrary channel counts.
        self.g = InvCNNNet(
            conf.n_layers,
            conf.im_shape[-1],
            hidden_chans=getattr(conf, 'hidden_chans', 128),
            data_channels=conf.im_shape[0],
        )
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

        # Larger fixed noise batch used only for quantitative classifier
        # metrics — 16 samples (val_batch_size) is too few for reliable
        # entropy / coverage. Lazy: only allocated when eval_classifier_path
        # is provided and the eval objects are loaded.
        self._eval_n = int(getattr(conf, 'eval_n_samples', 256))
        self.register_buffer("eval_z", torch.rand(self._eval_n, *conf.im_shape)*2.-1.)

        # Eval objects (loaded on first valid() if eval_classifier_path is set).
        self.eval_classifier = None      # MNISTClassifier or None
        self._eval_test_x = None         # held-out clean test batch [N, C, H, W]
        self._eval_test_y = None         # corresponding labels
        self._eval_disabled = False      # set True if loading fails, prevents retry spam

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

        # --- L_classifier (pretrained classifier-features perceptual loss):
        # match the penultimate-layer activations of a frozen MNIST classifier
        # for x and f(x). The classifier is the one already trained for
        # eval metrics (`python eval_metrics.py train ...`). Frozen + pretrained,
        # so unlike L_feat (which uses the trainable g), there's no degenerate
        # "saturate the feature extractor" escape — the target is fixed and
        # semantically meaningful.
        #
        # Why this addresses blur where L_rec doesn't: L1 in pixel space is
        # mean-seeking — when the model is uncertain about a pixel, the optimal
        # L1 prediction is the median of plausible values, which averaged over
        # a dataset = blurry. The classifier's intermediate features represent
        # *digit identity* rather than per-pixel intensity. A smudged digit has
        # features unlike any real-digit cluster; a sharp-but-slightly-shifted
        # digit has features close to the right cluster. So the classifier loss
        # rewards commitment to a specific identity, fighting the blur.
        #
        # Cost: one extra forward through the (small, ~100K param) classifier
        # per training step. Frozen, so its parameters don't update; only fx
        # carries gradient back to f.
        # Disabled by default with lambda_classifier=0; also a no-op if the
        # eval_classifier hasn't been loaded (no --eval_classifier_path).
        if self.conf.lambda_classifier > 0 and self.eval_classifier is not None:
            real_feats = self.eval_classifier(x, return_features=True)[1].detach()
            fake_feats = self.eval_classifier(fx, return_features=True)[1]
            loss_classifier = torch.nn.functional.l1_loss(fake_feats, real_feats)
        else:
            loss_classifier = torch.zeros((), device=x.device)

        total_loss = (self.conf.lambda_rec * loss_rec +
                      self.conf.lambda_sparse * loss_sparse +
                      self.conf.lambda_tight * loss_tight +
                      self.conf.lambda_denoise * loss_denoise +
                      self.conf.lambda_feat * loss_feat +
                      self.conf.lambda_classifier * loss_classifier)

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
        self.opt.step()


        return (total_loss.item(), loss_rec.item(), loss_sparse.item(),
                loss_tight.item(), loss_denoise.item(), loss_feat.item(),
                loss_classifier.item())

    def train_model(self, train_loader, n_epochs):
        self.opt = optim.Adam(self.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs)

        device = self.device()
        global_counter = 0

        # If lambda_classifier is enabled, ensure the classifier is loaded
        # before training starts (normally lazy-loaded in valid()).
        if getattr(self.conf, 'lambda_classifier', 0.0) > 0:
            self._maybe_load_eval()
            if self.eval_classifier is None:
                print("[WARN] --lambda_classifier > 0 but no classifier loaded "
                      "(check --eval_classifier_path). Classifier loss will be 0.")

        print("--- Starting Training for LinearIGN ---")
        for epoch in range(n_epochs):

            running_loss, running_rec, running_sparse, running_tight, running_denoise, running_feat, running_clf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            counter = 0
            num_batches = len(train_loader)

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                z = torch.randn_like(x)

                loss, loss_rec, loss_sparse, loss_tight, loss_denoise, loss_feat, loss_classifier = self.train_step(x, z)

                B = x.shape[0]
                counter += B
                global_counter += 1
                running_loss += loss * B
                running_rec += loss_rec * B
                running_sparse += loss_sparse * B
                running_tight += loss_tight * B
                running_denoise += loss_denoise * B
                running_feat += loss_feat * B
                running_clf += loss_classifier * B

                if (global_counter) % self.conf.log_freq == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    avg_loss = running_loss / counter
                    avg_rec = running_rec / counter
                    avg_sparse = running_sparse / counter
                    avg_tight = running_tight / counter
                    avg_denoise = running_denoise / counter
                    avg_feat = running_feat / counter
                    avg_clf = running_clf / counter
                    # Diagnostics on A's learned diagonal
                    A_active = self.A.diag.sum().item()        # number of dims with binary value 1
                    A_total = self.A.diag.numel()
                    A_probs_mean = self.A.probs.mean().item()  # soft fraction (sigmoid before rounding)
                    print(f"[Train] Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{num_batches}] "
                          f"LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Rec: {avg_rec:.4f} | Sparse: {avg_sparse:.4f} | tight: {avg_tight:.4f} | denoise: {avg_denoise:.4f} | feat: {avg_feat:.4f} | clf: {avg_clf:.4f} | A_active: {A_active:.0f}/{A_total} | A_probs_mean: {A_probs_mean:.3f}")
                    metrics = {
                    'step': global_counter, 'epoch': epoch+1, 'lr': current_lr,
                    'loss': avg_loss, 'rec': avg_rec, 'sparse': avg_sparse, 'tight': avg_tight,
                    'denoise': avg_denoise, 'feat': avg_feat, 'classifier': avg_clf,
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
                            'loss_classifier': avg_clf,
                            'A_active': A_active,
                            'A_probs_mean': A_probs_mean
                        }, step=global_counter)
                    counter = 0
                    running_loss, running_rec, running_sparse, running_tight, running_denoise, running_feat, running_clf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            if (epoch + 1) % self.conf.val_freq == 0:
                self.valid(epoch)

            self.sched.step()
    
    def _maybe_load_eval(self):
        """Lazy-load the MNIST classifier and a held-out test batch for OOD eval.

        Called from valid(). Only loads if --eval_classifier_path was set and
        the file exists. Skips silently (and disables itself) if not, so the
        IGN can still train without quantitative eval if the classifier hasn't
        been pretrained yet.
        """
        if self.eval_classifier is not None or self._eval_disabled:
            return
        path = getattr(self.conf, 'eval_classifier_path', '') or ''
        if not path:
            self._eval_disabled = True
            return
        if not os.path.exists(path):
            print(f"[eval] WARNING: --eval_classifier_path='{path}' not found, "
                  f"skipping classifier metrics. Train one with "
                  f"`python eval_metrics.py train --out {path}`.")
            self._eval_disabled = True
            return

        try:
            from eval_metrics import make_classifier
            from data import get_data_loaders
        except ImportError as e:
            print(f"[eval] WARNING: eval_metrics import failed ({e}); disabling eval metrics.")
            self._eval_disabled = True
            return

        device = self.device()
        # Pick the classifier architecture matching this run's dataset. The .pth
        # file at --eval_classifier_path must have been trained for the same
        # dataset (via `python eval_metrics.py train --dataset <name>`) or the
        # load will fail with a shape mismatch.
        try:
            clf = make_classifier(self.conf.dataset).to(device)
        except ValueError as e:
            print(f"[eval] WARNING: {e} — disabling eval metrics.")
            self._eval_disabled = True
            return
        clf.load_state_dict(torch.load(path, map_location=device))
        clf.eval()
        # Important: assign via object.__setattr__ to bypass nn.Module's
        # automatic submodule registration. The classifier is a logically
        # separate frozen oracle — we don't want its weights ending up in
        # self.state_dict() (which would bloat IGN checkpoints and break
        # loading in contexts where the classifier isn't present).
        # We also keep self.eval_classifier accessible for read everywhere.
        object.__setattr__(self, 'eval_classifier', clf)

        # Grab a fixed test batch for OOD eval; reused every validation epoch
        # so OOD numbers are comparable across epochs.
        _, test_loader = get_data_loaders(
            self.conf.dataset,
            self.conf.batch_size,
            min(self._eval_n, 256),
            self.conf.orig_im_size,
            self.conf.target_im_size,
        )
        x_clean, y_true = next(iter(test_loader))
        # Truncate to self._eval_n samples for a consistent batch size
        x_clean = x_clean[:self._eval_n].to(device)
        y_true = y_true[:self._eval_n].to(device)
        self._eval_test_x = x_clean
        self._eval_test_y = y_true
        print(f"[eval] Loaded MNIST classifier from {path}; "
              f"OOD test batch shape={tuple(x_clean.shape)}.")

    @torch.no_grad()
    def valid(self, epoch):
        self.eval()
        print("\n--- Running Validation ---")
        # test_model_properties(self) # Can be slow, optional

        # Generate samples from fixed noise
        gen_samples = self(self.valid_z)

        # Create a grid of z vs f(z). The denormalize function depends on the
        # dataset's normalization stats (MNIST vs CIFAR have different mean/std).
        denorm = get_denormalize_fn(self.conf.dataset)
        combined_grid = denorm(gen_samples).clip(0, 1)
        grid = make_grid(combined_grid, nrow=self.conf.val_batch_size)

        path = os.path.join(self.conf.grid_dir, f"samples_e{epoch+1}.png")
        imwrite(grid, path)
        print(f"Saved validation samples to {path}")

        if self.conf.wandb:
            wandb.log({"Samples (z vs f(z))": wandb.Image(path), "epoch": epoch})

        # Quantitative metrics — classifier-based generation eval + OOD projection
        # eval. Only runs if --eval_classifier_path was provided and loaded
        # successfully. Computed every validation epoch.
        self._maybe_load_eval()
        if self.eval_classifier is not None:
            from eval_metrics import classifier_metrics, ood_projection_metrics
            # Generation metrics on a larger fixed-noise batch (256 samples by
            # default, vs 16 for the visualization grid).
            gen_eval_samples = self(self.eval_z)
            gen_m = classifier_metrics(gen_eval_samples, self.eval_classifier)
            # OOD projection metrics on held-out test batch + Gaussian noise.
            ood_m = ood_projection_metrics(
                self, self.eval_classifier,
                self._eval_test_x, self._eval_test_y,
                noise_sigma=getattr(self.conf, 'noise_sigma', 0.3),
            )
            print(
                f"[eval] gen: entropy={gen_m['gen_entropy']:.3f} "
                f"confidence={gen_m['gen_confidence']:.3f} "
                f"coverage={gen_m['gen_coverage']}/10 | "
                f"ood: input_l1={ood_m['ood_input_l1']:.4f} "
                f"proj_l1={ood_m['ood_proj_l1']:.4f} "
                f"improv={ood_m['ood_improvement']:+.4f} "
                f"acc={ood_m['ood_class_acc']:.3f} "
                f"(clean_acc={ood_m['ood_clean_acc']:.3f})"
            )
            # Persist eval metrics to disk (one row per validation epoch),
            # alongside the WandB log. Lets you compare runs offline by reading
            # eval_metrics.csv from each run dir without WandB access.
            eval_row = {'epoch': epoch + 1, **gen_m, **ood_m}
            self.log_eval_metrics(eval_row)
            if self.conf.wandb:
                wandb.log({**gen_m, **ood_m, "epoch": epoch})

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
            # Drop eval_classifier.* keys before loading. Older checkpoints from
            # runs that had --eval_classifier_path set accidentally captured the
            # classifier as a submodule (nn.Module auto-registers child modules
            # on assignment), bloating the checkpoint and breaking strict loads
            # in test mode or runs with a different classifier path. The classifier
            # itself is loaded from --eval_classifier_path on demand, not from
            # the IGN checkpoint.
            classifier_keys = [k for k in ckpt if k.startswith('eval_classifier.')]
            if classifier_keys:
                print(f"  (dropping {len(classifier_keys)} stale eval_classifier.* keys)")
                ckpt = {k: v for k, v in ckpt.items() if not k.startswith('eval_classifier.')}
            # strict=False tolerates other minor mismatches (e.g. older checkpoints
            # missing newer buffers). Any real mismatches still get reported below.
            incompat = self.load_state_dict(ckpt, strict=False)
            if incompat.missing_keys or incompat.unexpected_keys:
                print(f"[load] partial load: "
                      f"{len(incompat.missing_keys)} missing, "
                      f"{len(incompat.unexpected_keys)} unexpected")
                if incompat.missing_keys:
                    print(f"[load]   missing (first 5):    {incompat.missing_keys[:5]}")
                if incompat.unexpected_keys:
                    print(f"[load]   unexpected (first 5): {incompat.unexpected_keys[:5]}")
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

    def log_eval_metrics(self, metrics):
        """Append a row to eval_metrics.csv (separate from the per-step metrics.csv).

        Eval metrics are computed once per validation epoch (different cadence
        from the per-step training metrics), so they live in their own file.
        Header is written on first call, rows appended thereafter.
        """
        log_path = os.path.join(self.conf.exp_dir, "eval_metrics.csv")
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
