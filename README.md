# 🧠 EEGSchizNet 

> **EEG-Based Schizophrenia Screening via Multi-Branch Deep Learning & Uncertainty Quantification**

*Can a neural network read the neural network inside your head?*

EEGSchizNet takes a 10-minute resting-state EEG recording and returns a clinical risk report — no expensive imaging, no specialist required at the point of screening. Built to be **honest**: subject-level evaluation only, no data leakage, calibrated uncertainty that knows when it doesn't know.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 82.1% (23/28 subjects) |
| **Sensitivity** (SCZ detection) | 85.7% — 12 of 14 SCZ correctly identified |
| **Specificity** (HC clearance) | 78.6% — 11 of 14 HC correctly cleared |
| **Evaluation** | 5-fold subject-stratified CV — zero data leakage |
| **Dataset** | REPOD (28 subjects, 14 SCZ, 14 HC) |

> **On honest evaluation:** Aich et al. 2025 report ~99% accuracy using segment-level cross-validation — the model sees the same subject in both train and test. Our 82.1% uses strict subject-stratified folds. Lower number. Real result.

---

## 🏗️ Architecture

```
19-Channel Raw EEG  (250 Hz · 4-second epochs · 1000 samples)
                         │
  ┌──────────────────────┼────────────────────┐
  │                      │                    │
  ▼                      ▼                    ▼
SpectralCNN        EEGConformer          PLI-GAT
CWT time-freq       Temporal dynamics     Phase connectivity
(ResBlocks x4)      + Microstate C        (Graph Attention)
256-dim             256-dim               256-dim
  │                   │                    │
  └───────────────────┴────────────────────┘
                 │
        Sigmoid Fusion Gate
         (learned weighting)
                 │
         256-dim embedding
                 │
      MC Dropout Classifier
       (30 stochastic passes)
                 │
    ┌────────────┼─────────────┬──────────────────┐
    ▼            ▼             ▼                  ▼
  Normal     Borderline    High Risk     Requires Further
 prob<0.30  prob 0.30-0.70  prob>0.70    Verification
                                         unc > 0.115
```

---

## 🔬 What Makes This Different

### Three Biologically Motivated Branches
Each branch captures a different signature of schizophrenia that the others cannot see:

- **SpectralCNN** — Elevated gamma power (interneuron dysfunction), elevated beta, reduced alpha. Uses CWT not FFT to preserve *when* spectral events happen, not just their average.
- **EEGConformer** — Long-range temporal dependencies via self-attention. Captures microstate C abnormalities: SCZ exits default mode network state faster, less frequently, with higher transition rate.
- **PLI-GAT** — Frontal-parietal disconnection in theta and alpha bands. Uses Phase Lag Index (not PLV) to avoid volume conduction artifact.

### Calibrated Uncertainty
30 MC Dropout passes per epoch. The model does not just give a probability — it tells you how stable that probability is. Two subjects with prob=0.60 but uncertainty 0.02 vs 0.25 require completely different clinical responses.

### Four Risk Zones
The **Requires Further Verification** zone is the key safety feature. When the model's 30 passes are inconsistent (std > 0.115), it refuses to auto-classify and flags the case for mandatory clinical review. A screening tool that cannot express uncertainty is dangerous.

### Interpretable Clinical Output
Every subject gets a 4-page PDF report with GradCAM saliency, PLI connectivity matrices, MC Dropout uncertainty analysis, and an 11-biomarker profile — translated into neuroscientifically interpretable evidence for clinicians.

---

## 🧬 The 11 Biomarkers

| Domain | Biomarker | SCZ Pattern |
|--------|-----------|-------------|
| EEG Power | Beta Power Ratio | Elevated |
| EEG Power | Gamma Power | Elevated |
| EEG Power | Theta/Alpha Ratio | Variable |
| EEG Power | LZ Complexity | Elevated |
| EEG Power | Hjorth Mobility | Elevated |
| PLI Connectivity | FP-PLI Theta | Reduced |
| PLI Connectivity | FP-PLI Alpha | Reduced |
| PLI Connectivity | Inter-Hemispheric PLI | Reduced |
| Microstate C | Duration | Shorter |
| Microstate C | Occurrence Rate | Less frequent |
| Microstate C | Transition Probability | Exits faster |

---

## 🚀 Pipeline

```bash
# Step 1: Preprocess raw EDF files
python preprocessing.py

# Step 2: Compute CWT time-frequency features
python cwt_precompute.py

# Step 3: Compute PLI connectivity graphs
python plv_precompute.py

# Step 4: Compute microstate C features
python microstate_precompute.py

# Step 5: Train across 5 folds
python train.py --epochs 50 --batch 32 --lr 3e-4

# Step 6: Evaluate with MC Dropout
python evaluate.py

# Step 7: Generate clinical PDF reports
python explain.py

# Single subject report
python explain.py --subject s18 --fold 1
```

---

## 📁 Repository Structure

```
EEGSchizNet_v2/
├── preprocessing.py          # EDF loading, bandpass filter, epoch extraction
├── cwt_precompute.py         # CWT time-frequency computation
├── plv_precompute.py         # PLI connectivity graphs
├── microstate_precompute.py  # Microstate C features
├── dataset.py                # 5-fold subject-stratified splits, DataLoader
├── model.py                  # EEGSchizNetV2 architecture
├── train.py                  # Asymmetric loss training, checkpointing
├── evaluate.py               # MC Dropout inference, subject aggregation
├── explain.py                # GradCAM, biomarkers, clinical PDF reports
└── results/
    └── eval_report.json      # Full evaluation results
```

---

## Key Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sampling rate | 250 Hz | REPOD native — no resampling |
| Epoch length | 4 seconds | Resolves theta (4 Hz) without artifact accumulation |
| Bandpass | 0.5-45 Hz | Removes DC drift and EMG noise |
| SCZ loss weight | 2.0 | False negatives penalized 2x — missing SCZ is worse |
| Early stopping | patience=15 | Prevents overfitting on small dataset |
| MC passes | 30 | Empirically stable uncertainty estimate |
| UNC threshold | 0.115 | Flags genuinely uncertain cases for clinical review |
| Embed dim | 256 | Per-branch output dimension |

---

## Clinical Risk Zones

| Zone | Condition | Recommendation |
|------|-----------|----------------|
| Normal | prob < 0.30, unc < 0.115 | Routine monitoring |
| Borderline | prob 0.30-0.70, unc < 0.115 | Clinical review recommended |
| High Risk | prob > 0.70, unc < 0.115 | Urgent psychiatric referral |
| Requires Further Verification | unc >= 0.115 | Mandatory further assessment |

---

## Per-Fold Results

| Fold | Accuracy | Sensitivity | Specificity | Correct |
|------|----------|-------------|-------------|---------|
| Fold 1 | 83.3% | 66.7% | 100.0% | 5/6 |
| Fold 2 | 83.3% | 100.0% | 66.7% | 5/6 |
| Fold 3 | 80.0% | 66.7% | 100.0% | 4/5 |
| Fold 4 | 80.0% | 100.0% | 66.7% | 4/5 |
| Fold 5 | 83.3% | 100.0% | 66.7% | 5/6 |
| **Overall** | **82.1%** | **85.7%** | **78.6%** | **23/28** |

### Confusion Matrix

| | Predicted SCZ | Predicted HC |
|--|---------------|--------------|
| **Actual SCZ (14)** | TP = 12 | FN = 2 |
| **Actual HC (14)** | FP = 3 | TN = 11 |

---

## Requirements

```bash
pip install torch torchvision
pip install mne scipy numpy scikit-learn
pip install torch-geometric
pip install reportlab
pip install matplotlib
```

---

## Dataset

**REPOD** — Resting-state EEG in schizophrenia
28 subjects · 14 SCZ · 14 HC · 19 channels · 250 Hz · ~10-15 min per subject

Raw data not included in this repository.
Available at: PhysioNet REPOD

---

## Disclaimer

This is a **research prototype** built on 28 subjects.
It is **not validated for clinical use**.
All outputs must be reviewed by a qualified psychiatrist or neurologist.
Clinical deployment requires regulatory approval and prospective validation on hundreds of subjects across multiple sites.

---

## Roadmap

- External validation on MSU schizophrenia EEG dataset (84 subjects)
- Cross-attention fusion replacing sigmoid gate
- Data augmentation to expand training set
- Federated learning for multi-site training
- FHIR-compatible output for EHR integration
- Regulatory pathway assessment (FDA De Novo / CE IVD)

---

## References

- REPOD Dataset — PhysioNet
- EEG-Conformer — Song et al. 2022
- Graph Attention Networks — Velickovic et al. 2018
- MC Dropout — Gal & Ghahramani 2016
- EEG Microstates — Lehmann et al. 1998
- Phase Lag Index — Stam et al. 2007

---

*Built on a DGX A100. Trained on 28 brains. Reports generated for every one of them.*
