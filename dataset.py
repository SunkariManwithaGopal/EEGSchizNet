"""
dataset.py  —  EEGSchizNet v2 Phase 4A+4C
Returns (x_cwt, x_time, x_graph, x_micro, label)

  x_cwt   : (4, 64, 500)   CWT feature maps  [log_mag, sin_phase, cos_phase, mag_skip]
  x_time  : (19, 1000)     z-scored raw EEG
  x_graph : (4, 19, 19)    4-band PLI
  x_micro : (3,)            microstate C features  [duration, occurrence, transition]
  label   : scalar int      0=HC  1=SCZ
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

CACHE_DIR = "/home/jovyan/EEGSchizNet_v2/cache"
N_FOLDS   = 5


# ── normalisation helpers ─────────────────────────────────────────────────────

def zscore(x: torch.Tensor) -> torch.Tensor:
    mu  = x.mean()
    std = x.std() + 1e-6
    return (x - mu) / std


# ── dataset ───────────────────────────────────────────────────────────────────

class EEGSchizDataset(Dataset):
    """
    Parameters
    ----------
    indices   : array-like of int  — epoch indices into the cache tensors
    X         : (N, 19, 1000) float32 tensor  — raw EEG
    CWT       : (N, 4, 64, 500) float32 tensor or None
    PLI       : (N, 4, 19, 19) float32 tensor or None
    microstates : (N, 3) float32 tensor or None
    y         : (N,) int tensor
    no_graph  : bool  — zero-out graph branch input
    no_micro  : bool  — zero-out microstate branch input
    """
    def __init__(self, indices, X, CWT, PLI, microstates, y,
                 no_graph=False, no_micro=False):
        self.idx    = np.asarray(indices)
        self.X      = X
        self.CWT    = CWT
        self.PLI    = PLI
        self.MS     = microstates
        self.y      = y
        self.no_graph = no_graph
        self.no_micro = no_micro

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]

        # ── temporal branch: z-score raw EEG ─────────────────────────────────
        x_time = zscore(self.X[k])                          # (19, 1000)

        # ── spectral branch: CWT feature maps ────────────────────────────────
        if self.CWT is not None:
            x_cwt = self.CWT[k]                             # (4, 64, 500)  already normalised
        else:
            # fallback: zeros (should not happen in practice)
            x_cwt = torch.zeros(4, 64, 500, dtype=torch.float32)

        # ── graph branch: PLI ─────────────────────────────────────────────────
        if self.PLI is not None and not self.no_graph:
            x_graph = self.PLI[k]                           # (4, 19, 19)
        else:
            x_graph = torch.zeros(4, 19, 19, dtype=torch.float32)

        # ── microstate branch ─────────────────────────────────────────────────
        if self.MS is not None and not self.no_micro:
            x_micro = self.MS[k]                            # (3,)
        else:
            x_micro = torch.zeros(3, dtype=torch.float32)

        label = self.y[k].long()

        return x_cwt, x_time, x_graph, x_micro, label


# ── fold builder ──────────────────────────────────────────────────────────────

def build_folds(no_graph=False, no_micro=False, verbose=True):
    """
    Returns list of 5 dicts, each with keys:
      'train_ds', 'val_ds', 'train_idx', 'val_idx',
      'has_graph', 'has_micro'
    """
    # load tensors
    y      = torch.load(os.path.join(CACHE_DIR, "y.pt"),      map_location="cpu")
    groups = torch.load(os.path.join(CACHE_DIR, "groups.pt"), map_location="cpu")
    X      = torch.load(os.path.join(CACHE_DIR, "X.pt"),      map_location="cpu")

    # CWT
    cwt_path = os.path.join(CACHE_DIR, "CWT.pt")
    if os.path.exists(cwt_path):
        CWT = torch.load(cwt_path, map_location="cpu")
        has_cwt = True
    else:
        CWT = None
        has_cwt = False
        if verbose:
            print("  CWT.pt        : NOT FOUND — zeros (run cwt_precompute.py)")

    # PLI
    pli_path = os.path.join(CACHE_DIR, "PLI.pt")
    if os.path.exists(pli_path):
        PLI = torch.load(pli_path, map_location="cpu")
        has_graph = True
    else:
        PLI = None
        has_graph = False

    # microstates
    ms_path = os.path.join(CACHE_DIR, "microstates.pt")
    if os.path.exists(ms_path):
        MS = torch.load(ms_path, map_location="cpu")
        has_micro = True
    else:
        MS = None
        has_micro = False

    if verbose:
        if has_cwt:
            print(f"  CWT.pt        : loaded {tuple(CWT.shape)}")
        if has_graph:
            print(f"  PLI.pt        : loaded {tuple(PLI.shape)}")
        if has_micro:
            print(f"  microstates.pt: loaded {tuple(MS.shape)}")
        else:
            print("  microstates.pt: NOT FOUND — zeros (run microstate_precompute.py)")

    y_np = y.numpy()
    g_np = groups.numpy()
    idx  = np.arange(len(y_np))

    sgkf  = StratifiedGroupKFold(n_splits=N_FOLDS)
    folds = []

    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(idx, y_np, g_np)):
        g_flag  = "yes" if (has_graph and not no_graph) else "no"
        m_flag  = "yes" if (has_micro and not no_micro) else "no"

        if verbose:
            n_tr_subj = len(np.unique(g_np[tr_idx]))
            n_va_subj = len(np.unique(g_np[va_idx]))
            print(f"  Fold {fold_i+1}: train={len(tr_idx):,} ({n_tr_subj} subj)"
                  f"  val={len(va_idx):,} ({n_va_subj} subj)"
                  f"  graph={g_flag}  micro={m_flag}")

        train_ds = EEGSchizDataset(
            tr_idx, X, CWT, PLI, MS, y,
            no_graph=no_graph, no_micro=no_micro
        )
        val_ds = EEGSchizDataset(
            va_idx, X, CWT, PLI, MS, y,
            no_graph=no_graph, no_micro=no_micro
        )

        folds.append({
            "train_ds" : train_ds,
            "val_ds"   : val_ds,
            "train_idx": tr_idx,
            "val_idx"  : va_idx,
            "has_graph": has_graph and not no_graph,
            "has_micro": has_micro and not no_micro,
        })

    return folds


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building folds …")
    folds = build_folds()
    print()
    ds = folds[0]["train_ds"]
    x_cwt, x_time, x_graph, x_micro, label = ds[0]
    print(f"  x_cwt   {tuple(x_cwt.shape)}   ← [log_mag, sin_phase, cos_phase, mag_skip]")
    print(f"  x_time  {tuple(x_time.shape)}")
    print(f"  x_graph {tuple(x_graph.shape)}")
    print(f"  x_micro {tuple(x_micro.shape)}  ← [duration, occurrence, transition]")
    print(f"  x_micro sample: {x_micro.tolist()}")
    print(f"  label   {label.item()}")