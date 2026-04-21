import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Import your existing components
from dataset import get_folds, to_spectrogram
from model import build_model

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("/home/jovyan/EEGSchizNet_v2/cache")
MODEL_SAVE_DIR = Path("/home/jovyan/EEGSchizNet_v2/models_aug")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 120          # Increased from 50
BATCH_SIZE = 32
PATIENCE = 15         # Early stopping: stop if no improvement after 15 epochs
LR = 1e-4             # Slightly lower for stability

# ── The Augmenter ──────────────────────────────────────────────────────────────
class EEGAugmenter:
    """Applies random transformations to raw EEG signal (19, 1000)"""
    def __init__(self, p=0.6):
        self.p = p

    def apply(self, x):
        if np.random.rand() > self.p:
            return x
        
        # Select one or more random augmentations
        choice = np.random.choice(['noise', 'scale', 'shift', 'all'])
        
        if choice in ['noise', 'all']:
            # Gaussian Noise (standard EEG sensor noise)
            x = x + np.random.normal(0, 0.01, x.shape)
            
        if choice in ['scale', 'all']:
            # Scaling (simulates skull thickness/impedance differences)
            x = x * np.random.uniform(0.8, 1.2)
            
        if choice in ['shift', 'all']:
            # Time Shifting (the feature doesn't always start at t=0)
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)
            
        return x

# ── New Dataset Wrapper ────────────────────────────────────────────────────────
class AugmentedDataset(Dataset):
    def __init__(self, indices, X_raw, y, augment=False):
        self.indices = indices
        self.X_raw = X_raw
        self.y = y
        self.augment = augment
        self.augmenter = EEGAugmenter()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        epoch = self.X_raw[idx].numpy()
        label = self.y[idx]

        if self.augment:
            epoch = self.augmenter.apply(epoch)

        # Use your existing to_spectrogram function
        spec = to_spectrogram(epoch)
        return torch.tensor(spec), label

# ── Training Logic with Early Stopping ─────────────────────────────────────────
def run_experiment():
    print("🚀 Starting Augmentation Experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load original data splits
    folds, X_all, y_all, g_all = get_folds()

    for fold_idx, (orig_train, orig_val) in enumerate(folds, 1):
        print(f"\n--- Training Fold {fold_idx} (with Augmentation) ---")
        
        # Wrap the original indices in our new AugmentedDataset
        train_ds = AugmentedDataset(orig_train.indices, X_all, y_all, augment=True)
        val_ds = AugmentedDataset(orig_val.indices, X_all, y_all, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = build_model(dropout_p=0.5).to(device) # Slightly higher dropout
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0
        epochs_no_improve = 0
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = torch.sigmoid(model(batch_x.to(device)))
                    all_probs.extend(out.cpu().numpy())
                    all_labels.extend(batch_y.numpy())
            
            val_auc = roc_auc_score(all_labels, all_probs)
            
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_SAVE_DIR / f"aug_fold{fold_idx}.pt")
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | Val AUC: {val_auc:.4f} | Best: {best_auc:.4f}")

            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
                break

if __name__ == "__main__":
    run_experiment()