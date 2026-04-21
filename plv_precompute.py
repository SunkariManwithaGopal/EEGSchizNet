"""
EEGSchizNet v2 — plv_precompute.py
===================================
Computes per-epoch Phase Lag Index (PLI) connectivity matrices from cached
raw EEG and saves them as a single tensor for fast training-time loading.

Run once, before training Phase 3:
  python plv_precompute.py

Output
------
  ~/EEGSchizNet_v2/cache/PLI.pt   — torch.FloatTensor (N, 19, 19)
  ~/EEGSchizNet_v2/cache/PLI_band_stats.json — per-band mean/std for QC

Why PLI and not PLV or coherence?
----------------------------------
Phase Lag Index discards zero-lag synchrony, which makes it immune to
volume conduction — the dominant confound in EEG connectivity. Two
electrodes picking up the same source will always have zero phase lag
regardless of their functional relationship; PLI discards exactly that.
Frontal PLI reduction in the theta/alpha band is one of the most
replicated schizophrenia EEG findings (Uhlhaas & Singer, 2010;
Krishnan et al., 2018) and is the biomarker underpinning Phase 3.

Band decomposition
------------------
We compute PLI in four canonical bands and concatenate them into a
multi-band connectivity tensor:
  theta  : 4–8 Hz   (working memory, prefrontal–hippocampal coupling)
  alpha  : 8–13 Hz  (inhibitory synchrony, reduced in schizophrenia)
  beta   : 13–30 Hz (sensorimotor, cognitive control loops)
  gamma  : 30–45 Hz (high-frequency binding, strongly disrupted in SCZ)

Output shape per epoch: (4, 19, 19) — stored as (N, 4, 19, 19).
The diagonal is always 1.0 (self-PLI), upper/lower triangles are
symmetric. The GAT in model.py uses the upper triangle as edge weights.

Memory
------
6583 epochs × 4 bands × 19 × 19 × 4 bytes ≈ 36 MB — well within budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert

CACHE_DIR = Path("/home/jovyan/EEGSchizNet_v2/cache")
SFREQ     = 250

BANDS = {
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_NAMES = list(BANDS.keys())   # order is canonical — do not change


# ── signal helpers ─────────────────────────────────────────────────────────────

def bandpass(data: np.ndarray, low: float, high: float,
             fs: float = SFREQ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. data: (..., samples)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def pli_matrix(epoch: np.ndarray) -> np.ndarray:
    """
    Compute PLI connectivity matrix for one epoch.

    Parameters
    ----------
    epoch : np.ndarray (19, samples)  — already bandpass-filtered

    Returns
    -------
    pli : np.ndarray (19, 19)  — symmetric, diagonal = 1.0
    """
    n_ch = epoch.shape[0]
    # Analytic signal via Hilbert transform
    analytic = hilbert(epoch, axis=-1)          # (19, samples) complex
    phase    = np.angle(analytic)               # (19, samples) float

    pli = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phase[i] - phase[j]          # instantaneous phase diff
            val  = float(np.abs(np.mean(np.sign(np.sin(dphi)))))
            pli[i, j] = val
            pli[j, i] = val
    np.fill_diagonal(pli, 1.0)
    return pli


def pli_matrix_fast(epoch: np.ndarray) -> np.ndarray:
    """
    Vectorised PLI — replaces the nested loop above.
    Same output, ~40× faster.

    epoch : (19, samples) bandpass-filtered
    """
    analytic = hilbert(epoch, axis=-1)                    # (19, T) complex
    phase    = np.angle(analytic)                         # (19, T)
    # pairwise phase difference via broadcasting: (19, 1, T) - (1, 19, T)
    dphi     = phase[:, None, :] - phase[None, :, :]     # (19, 19, T)
    pli      = np.abs(np.mean(np.sign(np.sin(dphi)), axis=-1)).astype(np.float32)
    np.fill_diagonal(pli, 1.0)
    return pli


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("EEGSchizNet v2 — PLI precompute")
    print(f"  Loading X.pt from {CACHE_DIR} …", flush=True)

    X = torch.load(CACHE_DIR / "X.pt")   # (N, 19, 1000)
    y = torch.load(CACHE_DIR / "y.pt")   # (N,)
    N = X.shape[0]
    print(f"  Epochs : {N}   Channels : {X.shape[1]}   Samples : {X.shape[2]}")
    print(f"  Bands  : {BAND_NAMES}")
    print(f"  Output : (N={N}, bands=4, ch=19, ch=19)")

    PLI_all = np.zeros((N, len(BANDS), 19, 19), dtype=np.float32)

    t0 = time.time()
    for i in range(N):
        epoch = X[i].numpy()   # (19, 1000)
        for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
            filtered     = bandpass(epoch, low, high)
            PLI_all[i, b_idx] = pli_matrix_fast(filtered)

        if (i + 1) % 500 == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (N - i - 1) / rate
            print(f"  [{i+1:>5}/{N}]  {elapsed:>6.1f}s elapsed  "
                  f"ETA {eta:>5.1f}s  ({rate:.1f} epochs/s)")

    print(f"\n  Total compute: {time.time()-t0:.1f}s")

    # ── QC: per-band mean PLI by class ────────────────────────────────────────
    y_np   = y.numpy()
    stats  = {}
    for b_idx, band_name in enumerate(BAND_NAMES):
        band_pli = PLI_all[:, b_idx, :, :]   # (N, 19, 19)
        # upper triangle only (excluding diagonal)
        idx = np.triu_indices(19, k=1)
        edges = band_pli[:, idx[0], idx[1]]  # (N, 171)

        hc_mean  = float(edges[y_np == 0].mean())
        scz_mean = float(edges[y_np == 1].mean())
        hc_std   = float(edges[y_np == 0].std())
        scz_std  = float(edges[y_np == 1].std())

        stats[band_name] = {
            "hc_mean":  round(hc_mean,  4),
            "scz_mean": round(scz_mean, 4),
            "hc_std":   round(hc_std,   4),
            "scz_std":  round(scz_std,  4),
            "delta":    round(hc_mean - scz_mean, 4),
        }
        print(f"  {band_name:>6}  HC={hc_mean:.4f}±{hc_std:.4f}  "
              f"SCZ={scz_mean:.4f}±{scz_std:.4f}  "
              f"Δ(HC−SCZ)={hc_mean-scz_mean:+.4f}")

    print()
    print("  ⚑  Check: Δ should be positive for theta/alpha/gamma")
    print("     (HC has higher PLI — stronger synchrony than SCZ)")

    # ── save ─────────────────────────────────────────────────────────────────
    pli_tensor = torch.tensor(PLI_all)
    torch.save(pli_tensor, CACHE_DIR / "PLI.pt")
    print(f"\n  PLI.pt saved  → shape {tuple(pli_tensor.shape)}  "
          f"dtype {pli_tensor.dtype}  "
          f"size {pli_tensor.numel()*4/1e6:.1f} MB")

    stats_path = CACHE_DIR / "PLI_band_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"bands": stats, "shape": list(pli_tensor.shape)}, f, indent=2)
    print(f"  PLI_band_stats.json saved → {stats_path}")
    print("\nplv_precompute.py done ✓")


if __name__ == "__main__":
    main()