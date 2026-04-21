# EEGSchizNet v2
EEG-Based Schizophrenia Screening via Multi-Branch Deep Learning & Uncertainty Quantification

## Results
- **Accuracy: 82.1%** (23/28 subjects correct)
- **Sensitivity: 85.7%** (12/14 SCZ correctly identified)
- **Specificity: 78.6%** (11/14 HC correctly cleared)
- Dataset: REPOD (28 subjects, 19-channel, 250Hz)
- Evaluation: 5-fold subject-stratified cross-validation (no data leakage)

## Architecture
- **Branch 1:** Spectral-Spatial CNN on CWT time-frequency representation
- **Branch 2:** EEG-Conformer with microstate C features
- **Branch 3:** PLI Graph Attention Network
- **Fusion:** Learned sigmoid fusion gate
- **Classifier:** MC Dropout (30 passes) with uncertainty quantification

## Risk Zones
- Normal (prob < 0.30)
- Borderline (prob 0.30-0.70)
- Requires Further Verification (uncertainty > 0.115)
- High Risk (prob > 0.70)

## Pipeline
1. `preprocessing.py` — bandpass filter, epoch extraction
2. `cwt_precompute.py` — CWT feature computation
3. `plv_precompute.py` — PLI connectivity graphs
4. `microstate_precompute.py` — microstate C features
5. `dataset.py` — 5-fold subject-stratified splits
6. `model.py` — EEGSchizNetV2 architecture
7. `train.py` — asymmetric loss training (SCZ weight=2.0)
8. `evaluate.py` — MC Dropout inference and biomarker computation
9. `explain.py` — GradCAM saliency and clinical PDF report generation

## Clinical Output
Per-subject 3-page PDF report including:
- Risk zone and recommendation
- Spectral GradCAM saliency map
- PLI connectivity matrices (4 bands)
- MC Dropout uncertainty analysis
- 11-biomarker profile (EEG power, PLI connectivity, microstate C)
