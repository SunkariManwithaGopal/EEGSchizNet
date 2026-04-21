import os
import numpy as np
import mne
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("/home/jovyan/dataset repod")
CACHE_DIR   = Path("/home/jovyan/EEGSchizNet_v2/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
SFREQ        = 250        # Hz
EPOCH_SEC    = 4          # seconds per window
EPOCH_SAMP   = SFREQ * EPOCH_SEC   # 1000 samples
BANDPASS_LOW = 0.5
BANDPASS_HI  = 45.0
NOTCH_FREQ   = 50.0
N_CHANNELS   = 19

CHANNELS = ['Fp2','F8','T4','T6','O2','Fp1','F7','T3','T5','O1',
            'F4','C4','P4','F3','C3','P3','Fz','Cz','Pz']

# ── helpers ────────────────────────────────────────────────────────────────────
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq, fs, quality=30):
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, data, axis=-1)

def process_subject(edf_path, label, subject_id):
    print(f"  Loading {edf_path.name} ...", end=" ", flush=True)

    # load
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # pick only our 19 channels (in fixed order)
    available = [ch for ch in CHANNELS if ch in raw.ch_names]
    if len(available) != N_CHANNELS:
        print(f"WARNING: only {len(available)} channels found, skipping.")
        return [], []
    raw.pick_channels(available, ordered=True)

    # get numpy array  shape: (19, n_samples)
    data = raw.get_data()   # in Volts

    # average reference
    data = data - data.mean(axis=0, keepdims=True)

    # bandpass
    data = bandpass_filter(data, BANDPASS_LOW, BANDPASS_HI, SFREQ)

    # notch
    data = notch_filter(data, NOTCH_FREQ, SFREQ)

    # epoch into non-overlapping 4-second windows
    n_epochs = data.shape[1] // EPOCH_SAMP
    epochs, labels = [], []
    for i in range(n_epochs):
        start = i * EPOCH_SAMP
        end   = start + EPOCH_SAMP
        epoch = data[:, start:end]   # (19, 1000)

        # basic artifact rejection: skip if any channel > 150 µV peak-to-peak
        if (epoch.max(axis=-1) - epoch.min(axis=-1)).max() > 150e-6:
            continue

        epochs.append(epoch.astype(np.float32))
        labels.append(label)

    print(f"{len(epochs)} epochs kept")
    return epochs, labels

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    all_epochs   = []   # list of (19, 1000) arrays
    all_labels   = []   # 0 = HC, 1 = SCZ
    all_subjects = []   # subject index (0–27) for group-aware CV

    subject_idx = 0
    subject_map = {}    # subject_id str → int index

    edf_files = sorted(DATASET_DIR.glob("*.edf"))
    print(f"Found {len(edf_files)} EDF files\n")

    for edf_path in edf_files:
        name = edf_path.stem   # e.g. "s01" or "h03"
        if name.startswith('s'):
            label = 1   # schizophrenia
        elif name.startswith('h'):
            label = 0   # healthy control
        else:
            print(f"Skipping unknown file: {name}")
            continue

        subject_map[name] = subject_idx
        epochs, labels = process_subject(edf_path, label, subject_idx)

        all_epochs.extend(epochs)
        all_labels.extend(labels)
        all_subjects.extend([subject_idx] * len(epochs))
        subject_idx += 1

    # convert to tensors
    X = torch.tensor(np.stack(all_epochs),   dtype=torch.float32)  # (N, 19, 1000)
    y = torch.tensor(all_labels,             dtype=torch.long)      # (N,)
    g = torch.tensor(all_subjects,           dtype=torch.long)      # (N,)  groups

    print(f"\n── Cache summary ──────────────────────────────")
    print(f"  Total epochs : {X.shape[0]}")
    print(f"  Shape        : {X.shape}   (epochs, channels, samples)")
    print(f"  SCZ epochs   : {(y==1).sum().item()}")
    print(f"  HC  epochs   : {(y==0).sum().item()}")
    print(f"  Subjects     : {subject_idx}")

    # save
    torch.save(X, CACHE_DIR / "X.pt")
    torch.save(y, CACHE_DIR / "y.pt")
    torch.save(g, CACHE_DIR / "groups.pt")

    # save subject map as text
    with open(CACHE_DIR / "subject_map.txt", "w") as f:
        for k, v in subject_map.items():
            f.write(f"{k} {v} {'SCZ' if k.startswith('s') else 'HC'}\n")

    print(f"\n  Saved to {CACHE_DIR}")
    print(f"  X.pt  → {X.shape}")
    print(f"  y.pt  → {y.shape}")
    print(f"  groups.pt → {g.shape}")
    print("\nPreprocessing done ✓")

if __name__ == "__main__":
    main()