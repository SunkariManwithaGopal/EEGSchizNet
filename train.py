"""
train.py  —  EEGSchizNet v2 Phase 4A+4B+4C  (retrain fix)
5-fold subject-stratified cross-validation.

Key changes from first run:
  - LR 3e-4 → 1e-4  (Conformer transformers overshoot at 3e-4)
  - OneCycleLR → linear warmup + cosine decay  (stable for transformer blocks)
  - patience 10 → 15, max_epochs 50 → 80  (CWT+Conformer converges slower)
  - Updated torch.amp API  (suppresses FutureWarnings)

Usage:
  python train.py                  # full training
  python train.py --folds 1        # smoke test
  python train.py --folds 1 --epochs 5
"""

import os, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from dataset import build_folds
from model   import EEGSchizNetV2

# ── config ────────────────────────────────────────────────────────────────────
MODELS_DIR    = "/home/jovyan/EEGSchizNet_v2/models"
LOGS_DIR      = "/home/jovyan/EEGSchizNet_v2/logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

LR            = 1e-4    # Conformer-safe peak LR
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 32
MAX_EPOCHS    = 80
PATIENCE      = 15
MIN_DELTA     = 1e-4
FN_WEIGHT     = 2.5    # matches PDF spec
WARMUP_EPOCHS = 5       # linear warmup before cosine decay
GRAD_CLIP     = 0.5     # tightened: transformer blocks produce extreme logits on hard folds
LABEL_SMOOTH  = 0.1    # prevents val loss explosion on high-confidence wrong predictions
NUM_WORKERS   = 4
PIN_MEMORY    = True


# ── asymmetric BCE with label smoothing ───────────────────────────────────────

class AsymmetricBCE(nn.Module):
    """
    BCEWithLogitsLoss + false-negative weighting + label smoothing.
    Label smoothing clips targets to [smooth/2, 1-smooth/2] so a confident
    wrong prediction produces finite bounded loss instead of exploding.
    Directly fixes val loss explosion in folds 3/4.
    """
    def __init__(self, fn_weight=2.0, label_smooth=0.1):
        super().__init__()
        self.fn_weight    = fn_weight
        self.label_smooth = label_smooth

    def forward(self, logits, targets):
        targets = targets.float()
        targets_smooth = targets * (1 - self.label_smooth) +                          (1 - targets) * self.label_smooth / 2
        loss   = F.binary_cross_entropy_with_logits(
                     logits, targets_smooth, reduction='none')
        weight = 1.0 + (self.fn_weight - 1.0) * targets
        return (loss * weight).mean()


# ── warmup + cosine LR schedule ───────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for warmup_epochs, then cosine decay to LR/100.
    Called once per epoch.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.01):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = base_lr * min_lr_ratio

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ── early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = None
        self.counter    = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        if self.best is None:
            self.best = value
            self.best_epoch = epoch
            return False, True
        improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            return self.counter >= self.patience, False


# ── train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for x_cwt, x_time, x_graph, x_micro, labels in loader:
        x_cwt   = x_cwt.to(device,   non_blocking=True)
        x_time  = x_time.to(device,  non_blocking=True)
        x_graph = x_graph.to(device, non_blocking=True)
        x_micro = x_micro.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(x_cwt, x_time, x_graph, x_micro).squeeze(1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n   = len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / n, auc


# ── validate one epoch ────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for x_cwt, x_time, x_graph, x_micro, labels in loader:
        x_cwt   = x_cwt.to(device,   non_blocking=True)
        x_time  = x_time.to(device,  non_blocking=True)
        x_graph = x_graph.to(device, non_blocking=True)
        x_micro = x_micro.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        with autocast('cuda'):
            logits = model(x_cwt, x_time, x_graph, x_micro).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n   = len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / n, auc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AsymmetricBCE(fn_weight=FN_WEIGHT, label_smooth=LABEL_SMOOTH)

    print("EEGSchizNet v2 — Phase 4A+4B+4C training  (retrain)")
    print(f"  device: {device}  |  amp: True")
    print(f"  epochs: {args.epochs}  |  batch: {BATCH_SIZE}  |  lr: {LR}")
    print(f"  patience: {PATIENCE}  |  warmup: {WARMUP_EPOCHS}ep"
          f"  |  fn_weight: {FN_WEIGHT}")
    print()
    print("Building folds …")
    folds = build_folds()
    print()

    log    = {}
    cv_auc = []

    for fold_i, fold in enumerate(folds):
        if fold_i >= args.folds:
            break

        k        = fold_i + 1
        train_ds = fold["train_ds"]
        val_ds   = fold["val_ds"]
        m_flag   = "yes" if fold["has_micro"] else "no"

        print("─" * 66)
        print(f"  Fold {k}  |  train={len(train_ds):,}  val={len(val_ds):,}")
        print(f"  graph={fold['has_graph']}  micro={m_flag}")
        print("─" * 66)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS,
                                  pin_memory=PIN_MEMORY, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=PIN_MEMORY)

        model     = EEGSchizNetV2().to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS,
            total_epochs=args.epochs, base_lr=LR)
        scaler    = GradScaler('cuda')
        es        = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

        best_state = None
        fold_log   = []

        print(f"  {'Ep':>4}  {'LR':>8}  {'TrLoss':>8}  {'TrAUC':>7}"
              f"  {'VaLoss':>8}  {'VaAUC':>7}  {'ES':>6}  {'Time':>6}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr = scheduler.step(epoch)
            tr_loss, tr_auc = train_epoch(
                model, train_loader, optimizer, criterion, scaler, device)
            va_loss, va_auc = val_epoch(model, val_loader, criterion, device)

            stop, improved = es.step(va_auc, epoch)
            elapsed = time.time() - t0

            star = " ✓" if improved else ""
            print(f"  {epoch:>4}  {lr:>8.2e}  {tr_loss:>8.4f}  {tr_auc:>7.4f}"
                  f"  {va_loss:>8.4f}  {va_auc:>7.4f}"
                  f"  {es.counter}/{PATIENCE}  {elapsed:>5.1f}s{star}")

            if improved:
                best_state = {k2: v.cpu().clone()
                              for k2, v in model.state_dict().items()}

            fold_log.append({
                "epoch": epoch, "lr": lr,
                "tr_loss": tr_loss, "tr_auc": tr_auc,
                "va_loss": va_loss, "va_auc": va_auc, "improved": improved,
            })

            if stop:
                print(f"\n  ⚑ Early stop @ ep{epoch}.  "
                      f"Best: ep{es.best_epoch} AUC={es.best:.4f}")
                break

        if best_state is None:
            best_state = {k2: v.cpu().clone()
                          for k2, v in model.state_dict().items()}

        ckpt_path = os.path.join(MODELS_DIR, f"fold{k}_best.pt")
        torch.save(best_state, ckpt_path)
        print(f"\n  Best val AUC fold {k}: {es.best:.4f}  →  fold{k}_best.pt\n")

        cv_auc.append(es.best)
        log[f"fold{k}"] = {
            "best_auc"  : es.best,
            "best_epoch": es.best_epoch,
            "history"   : fold_log,
        }

    print("═" * 66)
    print("  Cross-validation summary (Phase 4A+4B+4C  retrain)")
    for i, auc in enumerate(cv_auc):
        best_ep = log[f"fold{i+1}"]["best_epoch"]
        print(f"    Fold {i+1}: {auc:.4f}  [best @ep{best_ep}]")
    print(f"  Mean AUC : {np.mean(cv_auc):.4f}")
    print(f"  Std  AUC : {np.std(cv_auc):.4f}")
    print("═" * 66)

    log_path = os.path.join(LOGS_DIR, "train_log_4ABC.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log → {log_path}")


if __name__ == "__main__":
    main()