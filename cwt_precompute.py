"""
cwt_precompute.py  —  EEGSchizNet v2 Phase 4A
Computes Morlet CWT for every epoch and saves cache/CWT.pt

Output shape: (6583, 4, 64, 500)  float32
  dim 0 : epochs
  dim 1 : 4 feature maps  [log_magnitude, sin_phase, cos_phase, magnitude_skip]
  dim 2 : 64 log-spaced frequency scales  (~1–45 Hz)
  dim 3 : 500 time points  (downsampled 2× from 1000)

Runtime estimate: ~25–40 min on DGX A100 (CPU-bound, uses all cores via joblib)
Disk: ~1.0 GB
"""

import os, time, json
import numpy as np
import torch
import pywt
from joblib import Parallel, delayed

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = "/home/jovyan/EEGSchizNet_v2/cache"
X_PATH      = os.path.join(CACHE_DIR, "X.pt")
OUT_PATH    = os.path.join(CACHE_DIR, "CWT.pt")
META_PATH   = os.path.join(CACHE_DIR, "CWT_meta.json")

FS          = 250          # sampling frequency Hz
N_SCALES    = 64           # frequency axis resolution
F_MIN       = 1.0          # Hz
F_MAX       = 45.0         # Hz
WAVELET     = "cmor1.5-1.0"
T_OUT       = 500          # time axis after 2× downsample
N_JOBS      = -1           # use all CPU cores
BATCH_PRINT = 200          # progress every N epochs


def build_scales(fs: int, f_min: float, f_max: float, n: int) -> np.ndarray:
    """Convert frequency range to pywt scales for cmor wavelet."""
    # central frequency of cmor1.5-1.0
    fc = pywt.central_frequency(WAVELET)
    # scale = fc * fs / frequency  (log-spaced frequencies → log-spaced scales, reversed)
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n)
    scales = fc * fs / freqs
    return scales[::-1].copy()   # ascending scale = descending frequency


def process_epoch(x_epoch: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    x_epoch : (19, 1000)  float32
    returns  : (4, 64, 500) float32
      [0] log1p normalised magnitude
      [1] sin(phase)
      [2] cos(phase)
      [3] raw magnitude (skip connection)
    """
    # CWT per channel  → stack → (19, 64, 1000) complex
    cwt_channels = []
    for ch in range(x_epoch.shape[0]):
        coeffs, _ = pywt.cwt(x_epoch[ch], scales, WAVELET, sampling_period=1.0 / FS)
        cwt_channels.append(coeffs)          # (64, 1000) complex128
    cwt = np.stack(cwt_channels, axis=0)     # (19, 64, 1000)

    magnitude = np.abs(cwt)                  # (19, 64, 1000)
    phase     = np.angle(cwt)               # (19, 64, 1000)

    # average across channels  → (64, 1000)
    mag_mean   = magnitude.mean(axis=0)
    phase_mean = phase.mean(axis=0)

    # downsample time axis 2×  → (64, 500)
    mag_ds   = mag_mean[:, ::2]
    phase_ds = phase_mean[:, ::2]

    # feature map 0: log1p magnitude, z-scored per scale
    log_mag = np.log1p(mag_ds)
    mu  = log_mag.mean(axis=1, keepdims=True)
    std = log_mag.std(axis=1, keepdims=True) + 1e-6
    log_mag_z = (log_mag - mu) / std

    # feature maps 1 & 2: phase encoding
    sin_phase = np.sin(phase_ds)
    cos_phase = np.cos(phase_ds)

    # feature map 3: magnitude skip (global z-score)
    mag_skip = (mag_ds - mag_ds.mean()) / (mag_ds.std() + 1e-6)

    out = np.stack([log_mag_z, sin_phase, cos_phase, mag_skip], axis=0).astype(np.float32)
    return out   # (4, 64, 500)


def main():
    print("EEGSchizNet v2 — CWT precompute (Phase 4A)")
    print(f"  Loading X.pt from {CACHE_DIR} …")
    X = torch.load(X_PATH, map_location="cpu").numpy()   # (6583, 19, 1000)
    N, C, T = X.shape
    print(f"  Epochs : {N}   Channels : {C}   Samples : {T}")

    scales = build_scales(FS, F_MIN, F_MAX, N_SCALES)
    freqs  = pywt.scale2frequency(WAVELET, scales) * FS
    print(f"  Scales : {N_SCALES}   Freq range : {freqs.min():.2f}–{freqs.max():.2f} Hz")
    print(f"  Output : ({N}, 4, {N_SCALES}, {T_OUT})  float32")
    print(f"  Wavelet: {WAVELET}   Jobs: {N_JOBS}")
    print()

    t0 = time.time()

    # process in chunks to print progress
    chunk = BATCH_PRINT
    results = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch = Parallel(n_jobs=N_JOBS)(
            delayed(process_epoch)(X[i], scales) for i in range(start, end)
        )
        results.extend(batch)
        elapsed = time.time() - t0
        rate    = (end) / elapsed
        eta     = (N - end) / rate if rate > 0 else 0
        print(f"  [{end:5d}/{N}]  {elapsed:6.1f}s  ETA {eta:5.1f}s  ({rate:.0f} epochs/s)")

    print()
    print("  Stacking results …")
    CWT = np.stack(results, axis=0)   # (6583, 4, 64, 500)
    print(f"  CWT array shape : {CWT.shape}   dtype : {CWT.dtype}")

    # QC: check for NaN/Inf
    n_nan = np.isnan(CWT).sum()
    n_inf = np.isinf(CWT).sum()
    print(f"  QC — NaN: {n_nan}   Inf: {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("  ⚠ WARNING: NaN/Inf detected — check preprocessing")

    # QC: per-feature-map statistics
    for i, name in enumerate(["log_mag_z", "sin_phase", "cos_phase", "mag_skip"]):
        fm = CWT[:, i]
        print(f"  QC [{i}] {name:12s}: mean={fm.mean():.4f}  std={fm.std():.4f}"
              f"  min={fm.min():.4f}  max={fm.max():.4f}")

    print()
    print(f"  Saving CWT.pt → {OUT_PATH}")
    torch.save(torch.from_numpy(CWT), OUT_PATH)

    size_gb = os.path.getsize(OUT_PATH) / 1e9
    print(f"  File size : {size_gb:.2f} GB")

    meta = {
        "shape"   : list(CWT.shape),
        "dtype"   : "float32",
        "wavelet" : WAVELET,
        "n_scales": N_SCALES,
        "f_min"   : F_MIN,
        "f_max"   : F_MAX,
        "fs"      : FS,
        "t_out"   : T_OUT,
        "freqs_hz": freqs.tolist(),
        "scales"  : scales.tolist(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_PATH}")

    total = time.time() - t0
    print(f"\ncwt_precompute.py done ✓  ({total:.1f}s total)")


if __name__ == "__main__":
    main()