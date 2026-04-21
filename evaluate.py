"""
evaluate.py  —  EEGSchizNet v2  (Phase 4A+4B+4C  final)
MC Dropout evaluation with:
  - 30 stochastic passes
  - 3-zone clinical risk (Normal / Borderline / High Risk)
  - 11 biomarkers per subject across 3 sources

Biomarkers computed:
  From microstates.pt  (3):
    ms_duration        Microstate C mean duration (z-scored)
    ms_occurrence      Microstate C occurrence rate (z-scored)
    ms_transition      Microstate C transition probability (z-scored)

  From X.pt raw EEG    (5):
    theta_alpha_ratio  Frontal theta (4-8 Hz) / alpha (8-13 Hz) power ratio
    beta_power_ratio   Beta (13-30 Hz) relative power
    gamma_power        Gamma (30-45 Hz) relative power
    lz_complexity      Lempel-Ziv complexity (binarised signal)
    hjorth_mobility    Hjorth mobility parameter

  From PLI.pt          (3):
    pli_fp_theta       Frontal-parietal PLI, theta band
    pli_fp_alpha       Frontal-parietal PLI, alpha band
    pli_interhemi      Inter-hemispheric PLI mean (all bands)

Usage:
  python evaluate.py                          # full model
  python evaluate.py --no-micro              # ablate microstate branch
  python evaluate.py --no-graph --no-micro   # spectral+temporal only
  python evaluate.py --threshold 0.45        # adjust decision threshold
"""

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, precision_score)
from scipy.signal import welch

from dataset import build_folds
from model   import EEGSchizNetV2

CACHE_DIR   = "/home/jovyan/EEGSchizNet_v2/cache"
MODELS_DIR  = "/home/jovyan/EEGSchizNet_v2/models"
RESULTS_DIR = "/home/jovyan/EEGSchizNet_v2/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MC_PASSES   = 30          # matches PDF spec
BATCH_SIZE  = 64
NUM_WORKERS = 4
FS          = 250         # Hz

# ── 3-zone risk thresholds ────────────────────────────────────────────────────
# Normal      : prob < 0.35
# Borderline  : 0.35 ≤ prob < 0.65
# High Risk   : prob ≥ 0.65
ZONE_LOW  = 0.35
ZONE_HIGH = 0.65

# ── EEG channel indices (19-ch 10-20 order from preprocessing.py) ────────────
# ['Fp2','F8','T4','T6','O2','Fp1','F7','T3','T5','O1',
#  'F4','C4','P4','F3','C3','P3','Fz','Cz','Pz']
CH = {
    'Fp2':0,'F8':1,'T4':2,'T6':3,'O2':4,
    'Fp1':5,'F7':6,'T3':7,'T5':8,'O1':9,
    'F4':10,'C4':11,'P4':12,'F3':13,'C3':14,'P3':15,
    'Fz':16,'Cz':17,'Pz':18,
}
# frontal channels for theta/alpha ratio
FRONTAL = [CH['F3'], CH['F4'], CH['Fz'], CH['Fp1'], CH['Fp2']]
# frontal-parietal pairs for PLI
FP_PAIRS = [(CH['F3'], CH['P3']), (CH['F4'], CH['P4']), (CH['Fz'], CH['Pz'])]
# left vs right hemispheres for inter-hemispheric PLI
LEFT_CH  = [CH['F7'], CH['T3'], CH['T5'], CH['F3'], CH['C3'], CH['P3']]
RIGHT_CH = [CH['F8'], CH['T4'], CH['T6'], CH['F4'], CH['C4'], CH['P4']]


# ─────────────────────────────────────────────────────────────────────────────
# BIOMARKER COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def bandpower(epoch_ch, fs, fmin, fmax):
    """Mean PSD in [fmin, fmax] Hz for one channel epoch (1D array)."""
    f, pxx = welch(epoch_ch, fs=fs, nperseg=min(256, len(epoch_ch)))
    mask = (f >= fmin) & (f <= fmax)
    total = pxx.sum()
    return pxx[mask].sum() / (total + 1e-10)


def lz_complexity(signal):
    """
    Lempel-Ziv complexity of a binarised 1-D signal.
    Binarise at median, compute LZ76 complexity, normalise by signal length.
    """
    bits = (signal > np.median(signal)).astype(int)
    s = ''.join(map(str, bits))
    n = len(s)
    c, l, i, k, k_max = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    # normalise
    return c / (n / np.log2(n + 1e-10) + 1e-10)


def hjorth_mobility(signal):
    """Hjorth mobility = std(diff(x)) / std(x)."""
    d = np.diff(signal)
    return np.std(d) / (np.std(signal) + 1e-10)


def compute_eeg_biomarkers(X_subj: np.ndarray) -> dict:
    """
    X_subj : (N_epochs, 19, 1000)
    Returns dict of 5 EEG biomarkers (scalar, mean over epochs and frontal channels).
    """
    n_ep = X_subj.shape[0]

    theta_alpha = []
    beta_rel    = []
    gamma_rel   = []
    lz_vals     = []
    hjorth_vals = []

    for ep in X_subj:
        # frontal theta/alpha
        t_a = np.mean([
            bandpower(ep[c], FS, 4, 8) / (bandpower(ep[c], FS, 8, 13) + 1e-10)
            for c in FRONTAL
        ])
        theta_alpha.append(t_a)

        # beta and gamma relative power (all channels)
        beta  = np.mean([bandpower(ep[c], FS, 13, 30) for c in range(19)])
        gamma = np.mean([bandpower(ep[c], FS, 30, 45) for c in range(19)])
        beta_rel.append(beta)
        gamma_rel.append(gamma)

        # LZ complexity and Hjorth (Cz channel — most central)
        cz = ep[CH['Cz']]
        lz_vals.append(lz_complexity(cz))
        hjorth_vals.append(hjorth_mobility(cz))

    return {
        "theta_alpha_ratio": float(np.mean(theta_alpha)),
        "beta_power_ratio" : float(np.mean(beta_rel)),
        "gamma_power"      : float(np.mean(gamma_rel)),
        "lz_complexity"    : float(np.mean(lz_vals)),
        "hjorth_mobility"  : float(np.mean(hjorth_vals)),
    }


def compute_pli_biomarkers(PLI_subj: np.ndarray) -> dict:
    """
    PLI_subj : (N_epochs, 4, 19, 19)  bands: theta=0, alpha=1, beta=2, gamma=3
    Returns dict of 3 PLI biomarkers.
    """
    # mean over epochs
    pli_mean = PLI_subj.mean(axis=0)   # (4, 19, 19)

    # frontal-parietal PLI: theta (band 0) and alpha (band 1)
    fp_theta = np.mean([pli_mean[0, f, p] for f, p in FP_PAIRS])
    fp_alpha = np.mean([pli_mean[1, f, p] for f, p in FP_PAIRS])

    # inter-hemispheric PLI: mean across all bands and L-R channel pairs
    interhemi = np.mean([
        pli_mean[:, l, r].mean()
        for l, r in zip(LEFT_CH, RIGHT_CH)
    ])

    return {
        "pli_fp_theta"  : float(fp_theta),
        "pli_fp_alpha"  : float(fp_alpha),
        "pli_interhemi" : float(interhemi),
    }


def compute_microstate_biomarkers(MS_subj: np.ndarray) -> dict:
    """
    MS_subj : (N_epochs, 3)  [duration_z, occurrence_z, transition_z]
    """
    m = MS_subj.mean(axis=0)
    return {
        "ms_duration"  : float(m[0]),
        "ms_occurrence": float(m[1]),
        "ms_transition": float(m[2]),
    }


def zscore_biomarkers_across_subjects(all_results: list) -> list:
    """
    Z-score each biomarker across subjects within a fold so values are
    comparable (deviation from fold mean).  Modifies in place.
    """
    keys = list(all_results[0]["biomarkers"].keys())
    for k in keys:
        vals = np.array([r["biomarkers"][k] for r in all_results])
        mu, sd = vals.mean(), vals.std() + 1e-10
        for r in all_results:
            r["biomarkers_z"][k] = float((r["biomarkers"][k] - mu) / sd)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# RISK ZONE
# ─────────────────────────────────────────────────────────────────────────────

def risk_zone(prob: float, uncertainty: float) -> str:
    """
    Three clinical risk zones.
    High uncertainty (>0.10) degrades High Risk → Borderline.
    """
    if prob >= ZONE_HIGH:
        if uncertainty > 0.10:
            return "Borderline"   # confident model would say High but is uncertain
        return "High Risk"
    elif prob >= ZONE_LOW:
        return "Borderline"
    else:
        return "Normal"


def recommendation(zone: str) -> str:
    return {
        "High Risk"  : "Refer urgent — specialist psychiatric assessment recommended",
        "Borderline" : "Borderline — repeat EEG in 3 months or refer for clinical review",
        "Normal"     : "Unlikely SCZ — routine follow-up",
    }[zone]


# ─────────────────────────────────────────────────────────────────────────────
# MC DROPOUT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_inference(model, loader, device, n_passes):
    model.eval()
    model.enable_mc_dropout()

    all_pass_probs = [[] for _ in range(n_passes)]
    all_labels     = []

    for batch in loader:
        x_cwt, x_time, x_graph, x_micro, labels = batch
        x_cwt   = x_cwt.to(device,   non_blocking=True)
        x_time  = x_time.to(device,  non_blocking=True)
        x_graph = x_graph.to(device, non_blocking=True)
        x_micro = x_micro.to(device, non_blocking=True)

        for p in range(n_passes):
            logits = model(x_cwt, x_time, x_graph, x_micro).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_pass_probs[p].extend(probs.tolist())

        all_labels.extend(labels.tolist())

    probs  = np.array(all_pass_probs).T   # (N, n_passes)
    labels = np.array(all_labels)
    return probs, labels


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_subjects(probs, labels, val_idx,
                       groups_global, X_all, PLI_all, MS_all,
                       threshold=0.5, n_passes=MC_PASSES):
    mean_prob   = probs.mean(axis=1)
    uncertainty = probs.std(axis=1)
    epoch_to_subj = groups_global[val_idx]

    subj_ids = np.unique(epoch_to_subj)
    results  = []

    for sid in subj_ids:
        mask   = epoch_to_subj == sid
        ep_idx = val_idx[mask]          # indices into full cache arrays

        s_prob = float(mean_prob[mask].mean())
        s_unc  = float(uncertainty[mask].mean())
        s_vote = int(s_prob >= threshold)
        s_true = int(labels[mask][0])
        zone   = risk_zone(s_prob, s_unc)
        rec    = recommendation(zone)

        # raw biomarkers
        bio = {}
        bio.update(compute_eeg_biomarkers(X_all[ep_idx].numpy()))
        bio.update(compute_pli_biomarkers(PLI_all[ep_idx].numpy()))
        bio.update(compute_microstate_biomarkers(MS_all[ep_idx].numpy()))

        results.append({
            "subject_id"  : int(sid),
            "true_label"  : s_true,
            "mean_prob"   : s_prob,
            "uncertainty" : s_unc,
            "vote"        : s_vote,
            "correct"     : bool(s_vote == s_true),
            "n_epochs"    : int(mask.sum()),
            "risk_zone"   : zone,
            "recommendation": rec,
            "biomarkers"  : bio,
            "biomarkers_z": {},      # filled by zscore_biomarkers_across_subjects
        })

    zscore_biomarkers_across_subjects(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def epoch_metrics(probs_mean, labels, threshold):
    preds = (probs_mean >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {
        "acc" : float(accuracy_score(labels, preds)),
        "auc" : float(roc_auc_score(labels, probs_mean)),
        "f1"  : float(f1_score(labels, preds, zero_division=0)),
        "sens": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "ppv" : float(precision_score(labels, preds, zero_division=0)),
        "cm"  : {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
    }


def subject_metrics(subj_results):
    votes = [r["vote"]       for r in subj_results]
    trues = [r["true_label"] for r in subj_results]
    probs = [r["mean_prob"]  for r in subj_results]
    tn, fp, fn, tp = confusion_matrix(trues, votes, labels=[0,1]).ravel()
    return {
        "vote_accuracy": float(accuracy_score(trues, votes)),
        "auc"          : float(roc_auc_score(trues, probs)) if len(set(trues))>1 else 1.0,
        "sensitivity"  : float(tp/(tp+fn)) if (tp+fn)>0 else 0.0,
        "specificity"  : float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-graph",  action="store_true")
    parser.add_argument("--no-micro",  action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-passes", type=int,   default=MC_PASSES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("EEGSchizNet v2 — evaluation (11 biomarkers · 3-zone risk · 30 MC passes)")
    print(f"  device={device}  mc_passes={args.mc_passes}  threshold={args.threshold}")
    print(f"  no_graph={args.no_graph}  no_micro={args.no_micro}")
    print()

    # load full cache arrays for biomarker computation
    print("Loading cache arrays …")
    X_all   = torch.load(os.path.join(CACHE_DIR, "X.pt"),   map_location="cpu")
    PLI_all = torch.load(os.path.join(CACHE_DIR, "PLI.pt"), map_location="cpu")
    ms_path = os.path.join(CACHE_DIR, "microstates.pt")
    MS_all  = torch.load(ms_path, map_location="cpu") if os.path.exists(ms_path) \
              else torch.zeros(X_all.shape[0], 3)

    groups_global = torch.load(
        os.path.join(CACHE_DIR, "groups.pt"), map_location="cpu").numpy()

    print("Rebuilding folds …")
    folds = build_folds(no_graph=args.no_graph, no_micro=args.no_micro)
    print()

    fold_ep_m, fold_subj_m = [], []
    all_fold_results = {}

    for fold_i, fold in enumerate(folds):
        k = fold_i + 1
        ckpt_path = os.path.join(MODELS_DIR, f"fold{k}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {k}: checkpoint not found — skipping")
            continue

        model = EEGSchizNetV2()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(device)

        # training log info
        log_path = os.path.join("/home/jovyan/EEGSchizNet_v2/logs",
                                "train_log_4ABC.json")
        best_ep_str = best_auc_str = "?"
        if os.path.exists(log_path):
            with open(log_path) as f:
                tlog = json.load(f)
            if f"fold{k}" in tlog:
                best_ep_str  = tlog[f"fold{k}"]["best_epoch"]
                best_auc_str = f"{tlog[f'fold{k}']['best_auc']:.4f}"

        val_ds  = fold["val_ds"]
        val_idx = fold["val_idx"]
        loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

        print(f"  ── Fold {k} {'─'*50}")
        print(f"  Checkpoint : ep{best_ep_str}  (train AUC={best_auc_str})")
        print(f"  Computing MC Dropout ({args.mc_passes} passes) …")

        probs, labels = mc_inference(model, loader, device, args.mc_passes)
        mean_prob = probs.mean(axis=1)

        ep_m = epoch_metrics(mean_prob, labels, args.threshold)

        print(f"  Computing biomarkers for {len(np.unique(groups_global[val_idx]))} subjects …")
        subj_res = aggregate_subjects(
            probs, labels, val_idx, groups_global,
            X_all, PLI_all, MS_all,
            threshold=args.threshold, n_passes=args.mc_passes)
        subj_m = subject_metrics(subj_res)

        # ── print fold summary ────────────────────────────────────────────────
        print(f"  Epoch  : acc={ep_m['acc']:.4f}  AUC={ep_m['auc']:.4f}"
              f"  F1={ep_m['f1']:.4f}  sens={ep_m['sens']:.4f}"
              f"  spec={ep_m['spec']:.4f}")
        print(f"  Subject: acc={subj_m['vote_accuracy']:.4f}"
              f"  AUC={subj_m['auc']:.4f}"
              f"  sens={subj_m['sensitivity']:.4f}"
              f"  spec={subj_m['specificity']:.4f}")
        cm = ep_m["cm"]
        print(f"  Epoch CM : TN={cm['tn']} FP={cm['fp']}"
              f" FN={cm['fn']} TP={cm['tp']}")

        print("  Per-subject:")
        for r in subj_res:
            mark  = "✓" if r["correct"] else "✗"
            lbl   = "SCZ" if r["true_label"] == 1 else "HC"
            sid   = f"s{r['subject_id']:02d}"
            zone  = r["risk_zone"]
            b     = r["biomarkers"]
            bz    = r["biomarkers_z"]
            print(f"    [{mark}] {sid} ({lbl})  prob={r['mean_prob']:.3f}"
                  f"  unc={r['uncertainty']:.3f}  zone={zone}")
            print(f"         EEG  θ/α={b['theta_alpha_ratio']:.3f}"
                  f"  β={b['beta_power_ratio']:.4f}"
                  f"  γ={b['gamma_power']:.4f}"
                  f"  LZ={b['lz_complexity']:.3f}"
                  f"  Hjorth={b['hjorth_mobility']:.3f}")
            print(f"         PLI  FP-θ={b['pli_fp_theta']:.3f}"
                  f"  FP-α={b['pli_fp_alpha']:.3f}"
                  f"  IH={b['pli_interhemi']:.3f}")
            print(f"         μSt  dur={b['ms_duration']:+.2f}"
                  f"  occ={b['ms_occurrence']:+.2f}"
                  f"  trans={b['ms_transition']:+.2f}"
                  f"  (z-scores)")
            print(f"         → {r['recommendation']}")

        fold_ep_m.append(ep_m)
        fold_subj_m.append(subj_m)
        all_fold_results[f"fold{k}"] = {
            "epoch_metrics"  : ep_m,
            "subject_metrics": subj_m,
            "subjects"       : subj_res,
        }

    # ── cross-fold summary ────────────────────────────────────────────────────
    n = len(fold_ep_m)
    if n == 0:
        print("No folds evaluated.")
        return

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\n{'═'*64}")
    print(f"  EEGSchizNet v2 — evaluation summary ({n} folds)")
    print(f"{'═'*64}")
    print("  Epoch-level")
    for key, name in [("acc","accuracy"),("auc","auc"),("f1","f1"),
                      ("sens","sensitivity"),("spec","specificity"),("ppv","ppv")]:
        mu, sd = ms([m[key] for m in fold_ep_m])
        print(f"    {name:12s}: {mu:.4f} ± {sd:.4f}")
    print()
    print("  Subject-level (majority vote)")
    for key in ["vote_accuracy","auc","sensitivity","specificity"]:
        mu, sd = ms([m[key] for m in fold_subj_m])
        print(f"    {key:15s}: {mu:.4f} ± {sd:.4f}")

    # risk zone distribution
    all_zones = []
    for fk in all_fold_results.values():
        all_zones.extend([s["risk_zone"] for s in fk["subjects"]])
    print()
    print("  Risk zone distribution (all subjects across folds)")
    for zone in ["Normal","Borderline","High Risk"]:
        c = all_zones.count(zone)
        print(f"    {zone:12s}: {c}/{len(all_zones)}")
    print(f"{'═'*64}")

    # ── save ─────────────────────────────────────────────────────────────────
    suffix = ""
    if args.no_graph and args.no_micro:
        suffix = "_no_graph_no_micro"
    elif args.no_micro:
        suffix = "_no_micro"
    elif args.no_graph:
        suffix = "_no_graph"

    report_path  = os.path.join(RESULTS_DIR, f"eval_report{suffix}.json")
    summary_path = os.path.join(RESULTS_DIR, f"eval_summary{suffix}.txt")

    with open(report_path, "w") as f:
        json.dump(all_fold_results, f, indent=2)

    with open(summary_path, "w") as f:
        f.write("EEGSchizNet v2 — evaluation summary\n")
        f.write(f"mc_passes={args.mc_passes}  threshold={args.threshold}\n\n")
        f.write("Epoch-level\n")
        for key, name in [("acc","accuracy"),("auc","auc"),("f1","f1"),
                          ("sens","sensitivity"),("spec","specificity"),("ppv","ppv")]:
            mu, sd = ms([m[key] for m in fold_ep_m])
            f.write(f"  {name}: {mu:.4f} ± {sd:.4f}\n")
        f.write("\nSubject-level\n")
        for key in ["vote_accuracy","auc","sensitivity","specificity"]:
            mu, sd = ms([m[key] for m in fold_subj_m])
            f.write(f"  {key}: {mu:.4f} ± {sd:.4f}\n")
        f.write("\nRisk zone distribution\n")
        for zone in ["Normal","Borderline","High Risk"]:
            c = all_zones.count(zone)
            f.write(f"  {zone}: {c}/{len(all_zones)}\n")

    print(f"\n  Report  → {report_path}")
    print(f"  Summary → {summary_path}")
    print(f"\n  NOTE: biomarker computation is CPU-bound (~2-4 min for 5 folds).")
    print(f"  JSON report contains all 11 biomarkers per subject for PDF report generation.")


if __name__ == "__main__":
    main()