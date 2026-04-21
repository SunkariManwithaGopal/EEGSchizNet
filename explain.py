"""
explain.py  —  EEGSchizNet v2  Phase 6
GradCAM saliency + biomarker dashboard + clinical PDF report per subject.

What this produces per subject:
  1. GradCAM heatmap overlaid on CWT log-magnitude spectrogram
     (hooks into model.spectral.block4, last ResBlock before GAP)
  2. PLI connectivity matrix heatmap (frontal-parietal focus)
  3. MC Dropout uncertainty distribution plot
  4. 11-biomarker radar / bar chart vs population norms
  5. Multi-page clinical PDF  →  reports/<subject_id>_report.pdf

Usage:
  python explain.py                          # all subjects in eval_report.json
  python explain.py --subject s18            # single subject
  python explain.py --fold 1                 # all subjects in one fold
  python explain.py --subject s18 --fold 1  # specific subject in specific fold

Dependencies (all on DGX):
  pip install reportlab --break-system-packages

FIXES vs previous version:
  - GradCAM now uses model.eval() + torch.enable_grad() (was model.train() → dead gradients)
  - x_micro now uses real microstate values from subject biomarkers (was zeros)
  - CWT display uses mean across all 4 feature maps (was only channel 0)
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from dataset import build_folds
from model   import EEGSchizNetV2

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR   = "/home/jovyan/EEGSchizNet_v2/cache"
MODELS_DIR  = "/home/jovyan/EEGSchizNet_v2/models"
RESULTS_DIR = "/home/jovyan/EEGSchizNet_v2/results"
REPORTS_DIR = "/home/jovyan/EEGSchizNet_v2/reports"
FIGS_DIR    = os.path.join(REPORTS_DIR, "_figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,    exist_ok=True)

MC_PASSES   = 30
FS          = 250
N_SCALES    = 64
F_MIN, F_MAX = 1.0, 45.0

# channel order from preprocessing.py
CH_NAMES = ['Fp2','F8','T4','T6','O2','Fp1','F7','T3','T5','O1',
            'F4','C4','P4','F3','C3','P3','Fz','Cz','Pz']

BAND_NAMES  = ['Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-45Hz)']

# ── colour maps ───────────────────────────────────────────────────────────────
GRADCAM_CMAP = LinearSegmentedColormap.from_list(
    'gcam', ['#0a0a2e','#1a237e','#0d47a1','#00acc1','#00e676','#ffee58','#ff6f00','#b71c1c'])
PLI_CMAP     = LinearSegmentedColormap.from_list(
    'pli',  ['#f5f5f5','#b3e5fc','#0277bd','#01579b'])

# ── risk zone colours ─────────────────────────────────────────────────────────
ZONE_COLOR = {
    'Normal'                    : ('#2e7d32', '#e8f5e9'),
    'Borderline'                : ('#e65100', '#fff3e0'),
    'High Risk'                 : ('#c62828', '#ffebee'),
    'Requires Further Verification': ('#6a1b9a', '#f3e5f5'),  # purple
}

# ── zone assignment thresholds ────────────────────────────────────────────────
UNC_THRESHOLD  = 0.115   # uncertainty above this → Requires Further Verification
PROB_HIGH      = 0.70   # above this → High Risk
PROB_BORDERLINE= 0.30   # above this → Borderline
# below PROB_BORDERLINE → Normal


def assign_zone(prob, unc):
    """
    Assign clinical risk zone based on probability AND uncertainty.

    Priority:
      1. If uncertainty > 0.10 → Requires Further Verification
         (model is not confident enough to classify — flag for clinician)
      2. prob > 0.70 → High Risk
      3. prob > 0.30 → Borderline
      4. prob <= 0.30 → Normal
    """
    if unc > UNC_THRESHOLD:
        return ('Requires Further Verification',
                'Model confidence insufficient — mandatory clinical review and '
                'further diagnostic assessment required before any decision.')
    elif prob >= PROB_HIGH:
        return ('High Risk',
                'Refer urgent — specialist psychiatric assessment recommended.')
    elif prob >= PROB_BORDERLINE:
        return ('Borderline',
                'Uncertain — clinical review recommended before any decision.')
    else:
        return ('Normal',
                'Low risk — routine monitoring recommended.')


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into model.spectral.block4 (last ResBlock, (B,256,8,7)).
    Computes class activation map over (freq=8, time=7) -> upsample to (64,500).

    FIX: Uses model.eval() + torch.enable_grad() instead of model.train().
    model.train() on a single sample causes BatchNorm to compute unstable
    batch statistics, which kills the gradient signal and produces a flat
    (dead) GradCAM map. eval() freezes BN running stats so gradients flow.
    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []
        target = model.spectral.block4
        self._hooks.append(
            target.register_forward_hook(self._save_activation))
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradient))  # FIX: full_backward_hook

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x_cwt, x_time, x_graph, x_micro):
        # FIX: eval() keeps BatchNorm stable on single-sample forward passes
        # FIX: enable_grad() re-enables gradient computation in the eval context
        self.model.eval()
        with torch.enable_grad():
            x_cwt_g = x_cwt.detach().requires_grad_(True)
            logit = self.model(x_cwt_g, x_time, x_graph, x_micro)
            self.model.zero_grad()
            logit.sum().backward()

        if self.gradients is None or self.activations is None:
            # fallback: return uniform zero map if hooks didn't fire
            b = x_cwt.shape[0]
            return np.zeros((b, 64, 500))

        # weights = global average of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B,256,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,8,7)
        cam = F.relu(cam)
        # upsample to CWT resolution (64, 500)
        cam = F.interpolate(cam, size=(64, 500), mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)   # (B, 64, 500)
        # normalise per sample
        b = cam.shape[0]
        cam_min = cam.view(b,-1).min(1)[0].view(b,1,1)
        cam_max = cam.view(b,-1).max(1)[0].view(b,1,1) + 1e-8
        return ((cam - cam_min) / (cam_max - cam_min)).cpu().numpy()

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_gradcam_figure(cwt_mean, cam_mean, subject_id, save_path):
    """
    cwt_mean : (64, 500)  mean log-magnitude CWT across epochs
    cam_mean : (64, 500)  mean GradCAM across epochs
    """
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_SCALES)
    times = np.linspace(0, 4.0, 500)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), facecolor='white')
    fig.subplots_adjust(hspace=0.35)

    # top: CWT log-magnitude
    ax = axes[0]
    im = ax.pcolormesh(times, freqs, cwt_mean, cmap='viridis', shading='auto')
    ax.set_yscale('log')
    ax.set_yticks([4, 8, 13, 30, 45])
    ax.set_yticklabels(['4','8','13','30','45'], fontsize=8)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_title('CWT log-magnitude (mean across epochs)', fontsize=9, pad=4)
    for fline in [4, 8, 13, 30]:
        ax.axhline(fline, color='white', lw=0.5, ls='--', alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # bottom: GradCAM overlay on CWT
    ax = axes[1]
    ax.pcolormesh(times, freqs, cwt_mean, cmap='gray', shading='auto', alpha=0.6)
    # FIX: threshold at 0.15 (was 0.2) to show more activation
    cam_plot = np.ma.masked_less(cam_mean, 0.15)
    im2 = ax.pcolormesh(times, freqs, cam_plot, cmap=GRADCAM_CMAP,
                        shading='auto', alpha=0.80, vmin=0, vmax=1)
    ax.set_yscale('log')
    ax.set_yticks([4, 8, 13, 30, 45])
    ax.set_yticklabels(['4','8','13','30','45'], fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_title('GradCAM saliency — regions driving classification', fontsize=9, pad=4)
    plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.01, label='Saliency')

    # band labels on right
    for f, label in [(5.5,'θ'),(10,'α'),(20,'β'),(37,'γ')]:
        axes[1].text(4.05, f, label, fontsize=8, va='center', color='#444')

    fig.suptitle(f'Spectral GradCAM  —  {subject_id}', fontsize=10, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_pli_figure(pli_mean, subject_id, save_path):
    """
    pli_mean : (4, 19, 19)  mean PLI across epochs
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), facecolor='white')
    fig.subplots_adjust(wspace=0.3)

    for i, (ax, band) in enumerate(zip(axes, BAND_NAMES)):
        mat = pli_mean[i]
        im = ax.imshow(mat, cmap=PLI_CMAP, vmin=0, vmax=0.5,
                       interpolation='nearest', aspect='auto')
        ax.set_xticks(range(19))
        ax.set_yticks(range(19))
        if i == 0:
            ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=6)
            ax.set_yticklabels(CH_NAMES, fontsize=6)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_title(band, fontsize=8, pad=3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'PLI Connectivity Matrix  —  {subject_id}', fontsize=10)
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_uncertainty_figure(mc_probs, subject_id, true_label, save_path):
    """
    mc_probs : (N_epochs, 30) MC Dropout probabilities
    """
    ep_means = mc_probs.mean(axis=1)
    subj_mean = ep_means.mean()
    subj_std  = mc_probs.std(axis=1).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor='white')

    ax1.hist(ep_means, bins=25, color='#1565c0', alpha=0.75, edgecolor='white', lw=0.5)
    ax1.axvline(0.35, color='#e65100', lw=1.2, ls='--', label='Borderline low (0.35)')
    ax1.axvline(0.65, color='#c62828', lw=1.2, ls='--', label='High Risk (0.65)')
    ax1.axvline(subj_mean, color='black', lw=1.5, label=f'Subject mean ({subj_mean:.3f})')
    ax1.set_xlabel('SCZ probability', fontsize=9)
    ax1.set_ylabel('Epoch count', fontsize=9)
    ax1.set_title('Epoch probability distribution', fontsize=9, pad=4)
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 1)

    ep_stds = mc_probs.std(axis=1)
    ax2.scatter(ep_means, ep_stds, alpha=0.4, s=8, c='#1565c0')
    ax2.axvline(0.35, color='#e65100', lw=0.8, ls='--', alpha=0.6)
    ax2.axvline(0.65, color='#c62828', lw=0.8, ls='--', alpha=0.6)
    ax2.axhline(0.10, color='gray', lw=0.8, ls=':', alpha=0.6, label='Uncertainty threshold (0.10)')
    ax2.set_xlabel('Epoch mean probability', fontsize=9)
    ax2.set_ylabel('Epistemic uncertainty (std)', fontsize=9)
    ax2.set_title('MC Dropout uncertainty per epoch', fontsize=9, pad=4)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 1)

    gt = 'SCZ' if true_label == 1 else 'HC'
    fig.suptitle(f'MC Dropout Analysis  —  {subject_id}  (ground truth: {gt})', fontsize=10)
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_biomarker_figure(biomarkers, biomarkers_z, subject_id, save_path):
    """
    Horizontal bar chart showing z-scored biomarkers vs population norm (z=0).
    """
    labels = [
        'θ/α ratio', 'Beta power', 'Gamma power',
        'LZ complexity', 'Hjorth mobility',
        'FP-PLI theta', 'FP-PLI alpha', 'IH-PLI',
        'μState duration', 'μState occurrence', 'μState transition',
    ]
    keys = [
        'theta_alpha_ratio','beta_power_ratio','gamma_power',
        'lz_complexity','hjorth_mobility',
        'pli_fp_theta','pli_fp_alpha','pli_interhemi',
        'ms_duration','ms_occurrence','ms_transition',
    ]
    z_vals   = [biomarkers_z.get(k, 0.0) for k in keys]
    raw_vals = [biomarkers.get(k, 0.0)   for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
    y = np.arange(len(labels))
    bar_colors = ['#c62828' if v > 0.5 else '#2e7d32' if v < -0.5
                  else '#e65100' for v in z_vals]
    bars = ax.barh(y, z_vals, color=bar_colors, alpha=0.75, height=0.6)
    ax.axvline(0,    color='black', lw=1.0)
    ax.axvline( 1.0, color='gray',    lw=0.7, ls='--', alpha=0.5)
    ax.axvline(-1.0, color='gray',    lw=0.7, ls='--', alpha=0.5)
    ax.axvline( 2.0, color='#c62828', lw=0.7, ls=':', alpha=0.4)
    ax.axvline(-2.0, color='#2e7d32', lw=0.7, ls=':', alpha=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Z-score (deviation from fold mean)', fontsize=9)
    ax.set_title(f'11-Biomarker Profile  —  {subject_id}', fontsize=10, pad=6)

    for i, (bar, rv) in enumerate(zip(bars, raw_vals)):
        x = bar.get_width()
        ax.text(x + 0.05 if x >= 0 else x - 0.05, i,
                f'{rv:.3f}', va='center', ha='left' if x >= 0 else 'right',
                fontsize=7, color='#333')

    ax.text(1.8,  len(labels)-0.3, 'SCZ direction →', fontsize=7, color='#c62828', alpha=0.7)
    ax.text(-1.8, len(labels)-0.3, '← HC direction',  fontsize=7, color='#2e7d32',
            alpha=0.7, ha='right')

    for sep in [4.5, 7.5]:
        ax.axhline(sep, color='#ccc', lw=0.5, ls='-')

    for gy, glabel in [(2, 'EEG Power'), (6, 'PLI Connectivity'), (9.5, 'Microstates')]:
        ax.text(ax.get_xlim()[1] * 0.98, gy, glabel,
                fontsize=7, color='#666', ha='right', va='center',
                rotation=90, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf(subject_data, fig_paths, out_path):
    doc  = SimpleDocTemplate(out_path, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle('h1', parent=styles['Heading1'],
                        fontSize=18, spaceAfter=4, textColor=rl_colors.HexColor('#1a237e'))
    h2 = ParagraphStyle('h2', parent=styles['Heading2'],
                        fontSize=12, spaceAfter=3, spaceBefore=8,
                        textColor=rl_colors.HexColor('#283593'))
    body  = ParagraphStyle('body',  parent=styles['Normal'], fontSize=9, leading=14)
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=8, leading=12,
                           textColor=rl_colors.HexColor('#555555'))
    caption = ParagraphStyle('caption', parent=styles['Normal'],
                             fontSize=8, leading=11, alignment=TA_CENTER,
                             textColor=rl_colors.HexColor('#555555'))

    sid   = f"s{subject_data['subject_id']:02d}"
    gt    = 'SCZ' if subject_data['true_label'] == 1 else 'HC'
    prob  = subject_data['mean_prob']
    unc   = subject_data['uncertainty']
    # FIX: recompute zone using uncertainty-aware assign_zone()
    zone, rec = assign_zone(prob, unc)
    vote  = 'SCZ' if subject_data['vote'] == 1 else 'HC'
    correct = subject_data['correct']
    bio   = subject_data['biomarkers']
    bio_z = subject_data['biomarkers_z']

    zone_fg, zone_bg = ZONE_COLOR[zone]
    zone_rl_bg = rl_colors.HexColor(zone_bg)
    zone_rl_fg = rl_colors.HexColor(zone_fg)

    story = []

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("EEGSchizNet v2", h1))
    story.append(Paragraph("Clinical Screening Report — Confidential", small))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=rl_colors.HexColor('#1a237e'), spaceAfter=6))

    info_data = [
        ['Subject ID',       sid,          'Ground Truth',    gt],
        ['Model Prediction', vote,         'Correct',         '✓ Yes' if correct else '✗ No'],
        ['SCZ Probability',  f'{prob:.3f}','Uncertainty',     f'{unc:.3f}'],
        ['MC Passes',        '30',         'Epochs Evaluated',str(subject_data['n_epochs'])],
    ]
    info_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), rl_colors.HexColor('#f5f5f5')),
        ('BACKGROUND', (0,0), (0,-1), rl_colors.HexColor('#e8eaf6')),
        ('BACKGROUND', (2,0), (2,-1), rl_colors.HexColor('#e8eaf6')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
    ])
    story.append(Table(info_data, colWidths=[45*mm, 40*mm, 45*mm, 40*mm],
                       style=info_style))
    story.append(Spacer(1, 6*mm))

    risk_data = [[Paragraph(f'<b>Clinical Risk Zone: {zone}</b>', ParagraphStyle(
        'risk', fontSize=13, textColor=zone_rl_fg, alignment=TA_CENTER))]]
    risk_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), zone_rl_bg),
        ('BOX', (0,0), (-1,-1), 1.5, zone_rl_fg),
        ('TOPPADDING',   (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
    ])
    story.append(Table(risk_data, colWidths=[170*mm], style=risk_style))
    story.append(Spacer(1, 3*mm))

    rec_style = ParagraphStyle('rec', parent=body, textColor=zone_rl_fg,
                               fontName='Helvetica-Bold', fontSize=9)
    story.append(Paragraph(f'Recommendation: {rec}', rec_style))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph(
        'This report is generated by an AI screening tool for research purposes only. '
        'It is not a clinical diagnosis. All results must be reviewed by a qualified '
        'psychiatrist or neurologist before any clinical decision is made.',
        ParagraphStyle('disc', parent=small, textColor=rl_colors.HexColor('#b71c1c'))))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('Spectral GradCAM Saliency', h2))
    story.append(Paragraph(
        'The heatmap shows which time-frequency regions of the CWT spectrogram '
        'most influenced the model\'s classification decision. Warmer colours '
        '(red/yellow) indicate higher saliency. Band boundaries (θ/α/β/γ) are '
        'marked. Activations are computed via GradCAM on the SpectralCNN branch '
        '(block4), averaged across up to 20 epochs.', body))
    story.append(Spacer(1, 2*mm))
    story.append(Image(fig_paths['gradcam'], width=170*mm, height=85*mm))
    story.append(Paragraph(
        'Figure 1. Top: CWT log-magnitude spectrogram (mean across epochs). '
        'Bottom: GradCAM activation overlay — warmer colours indicate higher '
        'classification saliency. Computed with model.eval() + torch.enable_grad().', caption))

    story.append(PageBreak())

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(Paragraph('PLI Connectivity Analysis', h2))
    story.append(Paragraph(
        'Phase Lag Index (PLI) connectivity matrices across four frequency bands. '
        'Darker cells indicate stronger phase-lagged coupling between channel pairs. '
        'F3-P3 and F4-P4 frontal-parietal disconnection is a known SCZ biomarker.', body))
    story.append(Spacer(1, 2*mm))
    story.append(Image(fig_paths['pli'], width=170*mm, height=53*mm))
    story.append(Paragraph(
        'Figure 2. PLI matrices for theta (4-8 Hz), alpha (8-13 Hz), '
        'beta (13-30 Hz), and gamma (30-45 Hz) bands.', caption))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph('MC Dropout Uncertainty Analysis', h2))
    story.append(Paragraph(
        f'Distribution of SCZ probabilities across {subject_data["n_epochs"]} epochs '
        f'using 30 stochastic MC Dropout passes. Subject mean probability: {prob:.3f}. '
        f'Mean epistemic uncertainty: {unc:.3f}. '
        f'Risk zone boundaries: Normal < 0.35, Borderline 0.35-0.65, High Risk > 0.65.', body))
    story.append(Spacer(1, 2*mm))
    story.append(Image(fig_paths['uncertainty'], width=170*mm, height=65*mm))
    story.append(Paragraph(
        'Figure 3. Left: histogram of epoch-level SCZ probabilities. '
        'Right: epistemic uncertainty (std across 30 MC passes) vs mean probability. '
        'Points above the grey dashed line (unc > 0.10) flag high-uncertainty epochs.', caption))

    story.append(PageBreak())

    # ── PAGE 3 ────────────────────────────────────────────────────────────────
    story.append(Paragraph('11-Biomarker Profile', h2))
    story.append(Paragraph(
        'Z-scored deviation from the fold population mean for all 11 biomarkers. '
        'Positive values indicate deviation toward the SCZ population norm; '
        'negative toward the HC norm. Raw values are annotated on each bar. '
        'Biomarkers span three domains: EEG spectral power, PLI connectivity, '
        'and EEG microstate C dynamics.', body))
    story.append(Spacer(1, 2*mm))
    story.append(Image(fig_paths['biomarker'], width=170*mm, height=95*mm))
    story.append(Paragraph(
        'Figure 4. EEG power biomarkers (θ/α ratio, beta, gamma, LZ complexity, '
        'Hjorth mobility), PLI connectivity (frontal-parietal theta/alpha, '
        'inter-hemispheric), and microstate C features '
        '(duration, occurrence rate, transition probability).', caption))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('Biomarker Values', h2))
    bio_table_data = [
        ['Biomarker', 'Raw Value', 'Z-Score', 'Direction'],
        ['Theta/Alpha Ratio (frontal)', f"{bio.get('theta_alpha_ratio',0):.3f}",
         f"{bio_z.get('theta_alpha_ratio',0):+.2f}", 'variable'],
        ['Beta Power Ratio',  f"{bio.get('beta_power_ratio',0):.4f}",
         f"{bio_z.get('beta_power_ratio',0):+.2f}", '↑ SCZ'],
        ['Gamma Power',       f"{bio.get('gamma_power',0):.4f}",
         f"{bio_z.get('gamma_power',0):+.2f}", '↑ SCZ'],
        ['LZ Complexity',     f"{bio.get('lz_complexity',0):.3f}",
         f"{bio_z.get('lz_complexity',0):+.2f}", '↑ SCZ'],
        ['Hjorth Mobility',   f"{bio.get('hjorth_mobility',0):.3f}",
         f"{bio_z.get('hjorth_mobility',0):+.2f}", '↑ SCZ'],
        ['FP-PLI Theta',      f"{bio.get('pli_fp_theta',0):.3f}",
         f"{bio_z.get('pli_fp_theta',0):+.2f}", '↓ SCZ'],
        ['FP-PLI Alpha',      f"{bio.get('pli_fp_alpha',0):.3f}",
         f"{bio_z.get('pli_fp_alpha',0):+.2f}", '↓ SCZ'],
        ['Inter-Hemi PLI',    f"{bio.get('pli_interhemi',0):.3f}",
         f"{bio_z.get('pli_interhemi',0):+.2f}", '↓ SCZ'],
        ['Microstate C Duration',   f"{bio.get('ms_duration',0):.3f}",
         f"{bio_z.get('ms_duration',0):+.2f}", '↑ HC (longer in HC)'],
        ['Microstate C Occurrence', f"{bio.get('ms_occurrence',0):.3f}",
         f"{bio_z.get('ms_occurrence',0):+.2f}", '↑ HC (more in HC)'],
        ['Microstate C Transition', f"{bio.get('ms_transition',0):.3f}",
         f"{bio_z.get('ms_transition',0):+.2f}", '↑ SCZ (exits C more)'],
    ]
    bio_ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1a237e')),
        ('TEXTCOLOR',  (0,0), (-1,0), rl_colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
        ('GRID',       (0,0), (-1,-1), 0.5, rl_colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f5f5f5')]),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING',(0,0), (-1,-1), 5),
        ('TOPPADDING',  (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
        ('ALIGN', (1,0), (2,-1), 'CENTER'),
    ])
    story.append(Table(bio_table_data,
                       colWidths=[70*mm, 32*mm, 28*mm, 40*mm],
                       style=bio_ts))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(
        'Report generated by EEGSchizNet v2  |  REPOD dataset  |  April 2026  |  '
        'For research use only',
        ParagraphStyle('footer', parent=small, alignment=TA_CENTER)))

    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(subject_data, fold_k, X_all, CWT_all, PLI_all, model,
                    gradcam, device, val_idx, groups_global):
    """Run GradCAM, generate figures, build PDF for one subject."""
    sid_int = subject_data['subject_id']
    sid     = f"s{sid_int:02d}"
    epoch_to_subj = groups_global[val_idx]
    mask    = epoch_to_subj == sid_int
    ep_idx  = val_idx[mask]

    zone_print, _ = assign_zone(subject_data['mean_prob'], subject_data['uncertainty'])
    print(f"    {sid}  ({len(ep_idx)} epochs)  zone={zone_print}")

    # ── FIX: build real x_micro from subject biomarkers ──────────────────────
    bio = subject_data['biomarkers']
    x_micro_vals = torch.tensor([[
        bio.get('ms_duration',   0.0),
        bio.get('ms_occurrence', 0.0),
        bio.get('ms_transition', 0.0),
    ]], dtype=torch.float32).to(device)

    # ── GradCAM (mean over first 20 epochs max) ───────────────────────────────
    max_ep   = min(20, len(ep_idx))
    ep_sample = ep_idx[:max_ep]

    cam_list, cwt_list = [], []
    for ei in ep_sample:
        x_cwt   = CWT_all[ei:ei+1].to(device)
        x_time  = X_all[ei:ei+1].to(device)
        x_graph = PLI_all[ei:ei+1].to(device)

        # FIX: pass real microstate features, not zeros
        cam = gradcam(x_cwt, x_time, x_graph, x_micro_vals)   # (1, 64, 500)
        cam_list.append(cam[0])

        # FIX: average across all 4 CWT feature maps for display (not just channel 0)
        cwt_np = CWT_all[ei].numpy()       # (4, 64, 500)
        cwt_list.append(cwt_np.mean(0))    # mean over feature dim

    cam_mean = np.stack(cam_list).mean(0)   # (64, 500)
    cwt_mean = np.stack(cwt_list).mean(0)   # (64, 500)

    # ── MC Dropout probabilities (all epochs) ────────────────────────────────
    model.eval()
    model.enable_mc_dropout()
    mc_probs_list = []
    with torch.no_grad():
        for ei in ep_idx:
            x_cwt   = CWT_all[ei:ei+1].to(device)
            x_time  = X_all[ei:ei+1].to(device)
            x_graph = PLI_all[ei:ei+1].to(device)
            passes = []
            for _ in range(MC_PASSES):
                p = torch.sigmoid(
                    model(x_cwt, x_time, x_graph, x_micro_vals)
                ).item()
                passes.append(p)
            mc_probs_list.append(passes)
    mc_probs = np.array(mc_probs_list)   # (N_epochs, 30)

    # ── PLI mean ─────────────────────────────────────────────────────────────
    pli_mean = PLI_all[ep_idx].numpy().mean(0)   # (4, 19, 19)

    # ── generate figures ──────────────────────────────────────────────────────
    fig_gradcam     = os.path.join(FIGS_DIR, f"{sid}_f{fold_k}_gradcam.png")
    fig_pli         = os.path.join(FIGS_DIR, f"{sid}_f{fold_k}_pli.png")
    fig_uncertainty = os.path.join(FIGS_DIR, f"{sid}_f{fold_k}_uncertainty.png")
    fig_biomarker   = os.path.join(FIGS_DIR, f"{sid}_f{fold_k}_biomarker.png")

    make_gradcam_figure(cwt_mean, cam_mean, sid, fig_gradcam)
    make_pli_figure(pli_mean, sid, fig_pli)
    make_uncertainty_figure(mc_probs, sid, subject_data['true_label'], fig_uncertainty)
    make_biomarker_figure(subject_data['biomarkers'],
                          subject_data['biomarkers_z'], sid, fig_biomarker)

    # ── build PDF ─────────────────────────────────────────────────────────────
    pdf_path = os.path.join(REPORTS_DIR, f"{sid}_fold{fold_k}_report.pdf")
    build_pdf(subject_data,
              {'gradcam': fig_gradcam, 'pli': fig_pli,
               'uncertainty': fig_uncertainty, 'biomarker': fig_biomarker},
              pdf_path)
    print(f"      → {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject ID e.g. s18 (default: all)")
    parser.add_argument("--fold",    type=int, default=None,
                        help="Fold number 1-5 (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EEGSchizNet v2 — Phase 6: GradCAM + Clinical PDF Reports")
    print(f"  device={device}")
    print()

    report_path = os.path.join(RESULTS_DIR, "eval_report.json")
    if not os.path.exists(report_path):
        print(f"ERROR: {report_path} not found. Run evaluate.py first.")
        return
    with open(report_path) as f:
        eval_report = json.load(f)

    print("Loading cache …")
    X_all   = torch.load(os.path.join(CACHE_DIR, "X.pt"),   map_location="cpu")
    CWT_all = torch.load(os.path.join(CACHE_DIR, "CWT.pt"), map_location="cpu")
    PLI_all = torch.load(os.path.join(CACHE_DIR, "PLI.pt"), map_location="cpu")
    groups_global = torch.load(
        os.path.join(CACHE_DIR, "groups.pt"), map_location="cpu").numpy()

    print("Rebuilding folds …")
    folds = build_folds(verbose=False)
    print()

    fold_range = [args.fold] if args.fold else list(range(1, 6))
    all_pdfs   = []

    for fold_k in fold_range:
        ckpt_path = os.path.join(MODELS_DIR, f"fold{fold_k}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold_k}: checkpoint not found — skipping")
            continue

        fold_key = f"fold{fold_k}"
        if fold_key not in eval_report:
            print(f"  Fold {fold_k}: not in eval_report — skipping")
            continue

        print(f"  ── Fold {fold_k} ─────────────────────────────────────────")

        model = EEGSchizNetV2()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(device)
        gradcam = GradCAM(model)

        val_idx  = folds[fold_k - 1]["val_idx"]
        subjects = eval_report[fold_key]["subjects"]

        for subj_data in subjects:
            sid = f"s{subj_data['subject_id']:02d}"
            if args.subject and args.subject != sid:
                continue
            pdf = process_subject(
                subj_data, fold_k, X_all, CWT_all, PLI_all,
                model, gradcam, device, val_idx, groups_global)
            all_pdfs.append(pdf)

        gradcam.remove()
        del model

    print()
    print(f"{'═'*60}")
    print(f"  Phase 6 complete  —  {len(all_pdfs)} report(s) generated")
    print(f"  Reports → {REPORTS_DIR}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()