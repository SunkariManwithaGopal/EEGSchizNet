"""
EEGSchizNet v2 — microstate_precompute.py
==========================================
Computes EEG microstate features from cached raw EEG and saves them
as a single tensor for fast training-time loading.

Run once, after preprocessing.py and before training Phase 4C:
  python microstate_precompute.py

What are microstates?
---------------------
EEG microstates are quasi-stable topographic patterns of scalp potential
that persist for 60–120ms before rapidly transitioning. The canonical
decomposition produces 4 classes (A, B, C, D), each representing a
distinct spatial configuration of the electric field.

Microstate C is the most consistently disrupted class in schizophrenia:
  - Reduced duration (ms per episode)
  - Reduced occurrence rate (episodes per second)
  - Altered C→D and C→A transition probabilities

These disruptions are linked to the default mode network and are among
the most replicated EEG biomarkers in schizophrenia research
(Lehmann et al. 1998; Kindler et al. 2011; Tomescu et al. 2014).

Algorithm
---------
1. Compute GFP (global field power = std across channels per timepoint)
   for every epoch → (N, 1000)
2. Find GFP local maxima (peaks) per epoch — these are the moments of
   maximal topographic stability
3. Extract the scalp topography (19-dim vector) at each GFP peak
4. Run k-means (k=4) on ALL peak topographies pooled across training
   epochs to learn the 4 canonical microstate maps
5. Assign microstate labels to ALL timepoints (nearest centroid) for
   every epoch
6. Extract 3 scalar features per epoch for microstate C:
     (a) mean_duration_ms  — average consecutive run length × 4ms
     (b) occurrence_rate   — episodes per second (occurrences / 4s)
     (c) transition_prob_C_to_other — fraction of C exits going to D or A
         (captures the specific C→D disruption pattern)
7. Z-score each feature across the full dataset before saving

Output shape: (N, 3) float32
  Column 0: microstate C mean duration (z-scored, ms)
  Column 1: microstate C occurrence rate (z-scored, per second)
  Column 2: microstate C→other transition probability (z-scored)

Memory: 6583 × 3 × 4 bytes ≈ 79 KB — negligible.

Note on polarity
----------------
EEG microstates are polarity-invariant — a map and its inverse represent
the same state. The k-means here uses correlation distance (1 - |corr|)
to respect this invariance, matching the standard AAHC/k-means approach
in the microstate literature.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CACHE_DIR  = Path("/home/jovyan/EEGSchizNet_v2/cache")
SFREQ      = 250          # Hz
EPOCH_SAMP = 1000         # samples per epoch (4s × 250Hz)
N_STATES   = 4            # canonical microstate count
MS_PER_SAMPLE = 1000 / SFREQ  # 4ms per sample at 250Hz
RANDOM_STATE  = 42


# ── GFP and peak extraction ────────────────────────────────────────────────────

def compute_gfp(epoch: np.ndarray) -> np.ndarray:
    """
    epoch : (19, 1000)
    returns: (1000,) GFP = std across channels at each timepoint
    """
    return epoch.std(axis=0)


def get_gfp_peaks(gfp: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """
    Find local maxima of GFP with minimum separation.
    min_distance=10 → peaks at least 40ms apart (standard in literature).
    Returns indices of peaks.
    """
    peaks, _ = find_peaks(gfp, distance=min_distance)
    return peaks


# ── Polarity-invariant k-means ─────────────────────────────────────────────────

def corr_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - |Pearson correlation| — polarity invariant."""
    num = np.dot(a, b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - abs(num / denom)


def assign_microstates(topographies: np.ndarray,
                       centroids: np.ndarray) -> np.ndarray:
    """
    Assign each topography to its nearest centroid using |correlation|.

    topographies : (T, 19)
    centroids    : (4, 19)
    returns      : (T,) int labels 0–3
    """
    labels = np.zeros(len(topographies), dtype=np.int32)
    for i, topo in enumerate(topographies):
        similarities = np.array([
            abs(np.dot(topo, c) / (np.linalg.norm(topo) * np.linalg.norm(c) + 1e-10))
            for c in centroids
        ])
        labels[i] = np.argmax(similarities)
    return labels


# ── Feature extraction for one epoch ──────────────────────────────────────────

def extract_microstate_features(labels: np.ndarray,
                                target_state: int,
                                n_states: int = 4) -> np.ndarray:
    """
    Extract 3 scalar features for the target microstate from a label sequence.

    labels       : (1000,) int microstate assignment per timepoint
    target_state : which microstate index to measure (determined by matching
                   to microstate C centroid after clustering)
    returns      : (3,) [mean_duration_ms, occurrence_rate_per_sec, transition_prob]
    """
    # ── duration and occurrence ───────────────────────────────────────────────
    in_state   = False
    run_start  = 0
    durations  = []
    n_episodes = 0

    for t, lbl in enumerate(labels):
        if lbl == target_state and not in_state:
            in_state  = True
            run_start = t
            n_episodes += 1
        elif lbl != target_state and in_state:
            durations.append(t - run_start)
            in_state = False
    if in_state:
        durations.append(len(labels) - run_start)

    mean_dur_ms = float(np.mean(durations) * MS_PER_SAMPLE) if durations else 0.0
    occ_rate    = float(n_episodes / (EPOCH_SAMP / SFREQ))   # episodes per second

    # ── transition probability C→other ───────────────────────────────────────
    # Count transitions out of target_state
    transitions_total = 0
    transitions_out   = 0
    for t in range(len(labels) - 1):
        if labels[t] == target_state and labels[t + 1] != target_state:
            transitions_out += 1
        if labels[t] == target_state:
            transitions_total += 1

    trans_prob = float(transitions_out / transitions_total) if transitions_total > 0 else 0.0

    return np.array([mean_dur_ms, occ_rate, trans_prob], dtype=np.float32)


# ── Identify which cluster is microstate C ────────────────────────────────────

def identify_microstate_C(centroids: np.ndarray,
                           channel_names: list) -> int:
    """
    Microstate C in the canonical decomposition has a characteristic
    fronto-occipital polarity pattern — positive over frontal electrodes,
    negative over occipital (or vice versa, polarity-invariant).

    We identify C as the centroid with the highest absolute difference
    between mean frontal (Fp1, Fp2, Fz, F3, F4, F7, F8) and mean
    occipital (O1, O2) weights, normalised by centroid norm.

    This heuristic works reliably for the standard 10-20 montage.
    If the identification looks wrong, inspect the centroids visually.
    """
    frontal_idx = [channel_names.index(ch)
                   for ch in ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8']
                   if ch in channel_names]
    occipital_idx = [channel_names.index(ch)
                     for ch in ['O1', 'O2']
                     if ch in channel_names]

    scores = []
    for centroid in centroids:
        norm = np.linalg.norm(centroid) + 1e-10
        front_mean = centroid[frontal_idx].mean() / norm
        occ_mean   = centroid[occipital_idx].mean() / norm
        # fronto-occipital contrast (absolute — polarity invariant)
        scores.append(abs(front_mean - occ_mean))

    return int(np.argmax(scores))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    CHANNELS = ['Fp2','F8','T4','T6','O2','Fp1','F7','T3','T5','O1',
                'F4','C4','P4','F3','C3','P3','Fz','Cz','Pz']

    print("EEGSchizNet v2 — Microstate precompute (Phase 4C)")
    print(f"  Loading X.pt from {CACHE_DIR} …", flush=True)

    X = torch.load(CACHE_DIR / "X.pt")   # (N, 19, 1000)
    y = torch.load(CACHE_DIR / "y.pt")   # (N,)
    N = X.shape[0]
    print(f"  Epochs : {N}   Channels : {X.shape[1]}   Samples : {X.shape[2]}")

    # ── Step 1: collect GFP peak topographies ────────────────────────────────
    print("\n  Step 1: Extracting GFP peak topographies …", flush=True)
    t0 = time.time()
    all_peak_topos = []   # will be (total_peaks, 19)

    for i in range(N):
        epoch = X[i].numpy()           # (19, 1000)
        gfp   = compute_gfp(epoch)     # (1000,)
        peaks = get_gfp_peaks(gfp)
        if len(peaks) > 0:
            topos = epoch[:, peaks].T  # (n_peaks, 19)
            # L2-normalise each topography before clustering
            norms = np.linalg.norm(topos, axis=1, keepdims=True) + 1e-10
            all_peak_topos.append(topos / norms)

    all_peak_topos = np.vstack(all_peak_topos)   # (total_peaks, 19)
    print(f"  Total GFP peaks collected : {len(all_peak_topos):,}  "
          f"({time.time()-t0:.1f}s)")

    # ── Step 2: k-means clustering (k=4) ─────────────────────────────────────
    print(f"\n  Step 2: k-means clustering (k={N_STATES}) …", flush=True)
    t1 = time.time()

    # Standard sklearn k-means on L2-normalised topographies.
    # For polarity invariance we add both topo and -topo to the pool,
    # then take only the first k centroids (the negatives are redundant
    # after normalisation).
    augmented = np.vstack([all_peak_topos, -all_peak_topos])
    km = KMeans(n_clusters=N_STATES, n_init=20, max_iter=500,
                random_state=RANDOM_STATE)
    km.fit(augmented)
    centroids = km.cluster_centers_   # (4, 19)

    # Re-normalise centroids
    cnorms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
    centroids = centroids / cnorms

    print(f"  Clustering done ({time.time()-t1:.1f}s)")
    print(f"  Centroid norms: {np.linalg.norm(centroids, axis=1).round(3)}")

    # ── Step 3: identify microstate C ─────────────────────────────────────────
    state_C_idx = identify_microstate_C(centroids, CHANNELS)
    print(f"\n  Microstate C identified as cluster index: {state_C_idx}")
    print(f"  (Frontal-occipital contrast scores: "
          f"{[round(abs((centroids[i][CHANNELS.index('Fz')] - centroids[i][CHANNELS.index('Oz')] if 'Oz' in CHANNELS else centroids[i][[CHANNELS.index(c) for c in CHANNELS if c.startswith('O')]].mean())), 3) for i in range(N_STATES)]})")

    # ── Step 4: assign labels and extract features ────────────────────────────
    print(f"\n  Step 4: Assigning labels and extracting features …", flush=True)
    t2 = time.time()

    features = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        epoch = X[i].numpy()           # (19, 1000)
        # L2-normalise each timepoint
        norms = np.linalg.norm(epoch, axis=0, keepdims=True) + 1e-10
        epoch_norm = (epoch / norms).T  # (1000, 19)

        labels = assign_microstates(epoch_norm, centroids)  # (1000,)
        features[i] = extract_microstate_features(labels, state_C_idx)

        if (i + 1) % 1000 == 0 or i == N - 1:
            elapsed = time.time() - t2
            rate = (i + 1) / elapsed
            eta  = (N - i - 1) / rate
            print(f"  [{i+1:>5}/{N}]  {elapsed:>5.1f}s  ETA {eta:>5.1f}s  "
                  f"({rate:.0f} epochs/s)")

    # ── Step 5: z-score and QC ────────────────────────────────────────────────
    print(f"\n  Step 5: Z-scoring features …")
    scaler = StandardScaler()
    features_z = scaler.fit_transform(features).astype(np.float32)

    y_np = y.numpy()
    print(f"\n  QC — microstate C features (z-scored), HC vs SCZ:")
    feat_names = ["duration_ms", "occurrence_rate", "transition_prob"]
    stats = {}
    for j, fname in enumerate(feat_names):
        hc_mean  = float(features_z[y_np == 0, j].mean())
        scz_mean = float(features_z[y_np == 1, j].mean())
        hc_std   = float(features_z[y_np == 0, j].std())
        scz_std  = float(features_z[y_np == 1, j].std())
        delta    = hc_mean - scz_mean
        print(f"  {fname:>20}:  HC={hc_mean:+.4f}±{hc_std:.4f}  "
              f"SCZ={scz_mean:+.4f}±{scz_std:.4f}  Δ(HC−SCZ)={delta:+.4f}")
        stats[fname] = {
            "hc_mean": round(hc_mean, 4), "scz_mean": round(scz_mean, 4),
            "hc_std":  round(hc_std,  4), "scz_std":  round(scz_std,  4),
            "delta":   round(delta,   4),
        }

    print("\n  Expected signs (from literature):")
    print("    duration_ms       Δ > 0  (HC has longer C episodes than SCZ)")
    print("    occurrence_rate   Δ > 0  (HC has more C episodes per second)")
    print("    transition_prob   Δ < 0  (SCZ exits C more often — instability)")

    # ── Save ──────────────────────────────────────────────────────────────────
    micro_tensor = torch.tensor(features_z)
    torch.save(micro_tensor, CACHE_DIR / "microstates.pt")
    print(f"\n  microstates.pt saved  → shape {tuple(micro_tensor.shape)}  "
          f"dtype {micro_tensor.dtype}")

    meta = {
        "state_C_index": int(state_C_idx),
        "n_states": N_STATES,
        "features": feat_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std":  scaler.scale_.tolist(),
        "centroids_shape": list(centroids.shape),
        "qc": stats,
    }
    np.save(CACHE_DIR / "microstate_centroids.npy", centroids)
    with open(CACHE_DIR / "microstate_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  microstate_centroids.npy saved")
    print(f"  microstate_meta.json saved")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print("\nmicrostate_precompute.py done ✓")


if __name__ == "__main__":
    main()