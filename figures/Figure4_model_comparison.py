# -*- coding: utf-8 -*-
"""
Figure 3: Case-dependent model ranking under strict OOD
Data: authoritative CSVs from paper_evidence_20260414/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"

# ── Authoritative data (mean R² over 3 seeds) ────────────────────────────────
# Models shown per case: EKG, LinearFusion, CrossAttn, HGT, FastGTN
MODEL_LABELS = ['EKG', 'Linear\nFusion', 'Cross-\nAttn', 'HGT', 'FastGTN']
BAR_COLORS   = ['#E8A14A', '#C04D36', '#7C2E36', '#A68A5A', '#F2D69C']

CASES = {
    'A': {
        'title': 'Case A: Forest Biomass',
        'subtitle': '(150 km strict OOD)',
        'vals': [-0.0086, 0.0072, -0.2329, 0.0503, 0.1061],
        'stds': [ 0.0129, 0.0739,  0.6197, 0.0386, 0.0092],
        'ylim': (-1.0, 0.4),
        'annotation': ('High instability', 2, 'top'),   # (text, bar_idx_anchor, va)
        'best_label': None,
    },
    'B': {
        'title': 'Case B: California Housing',
        'subtitle': '(240 km strict OOD)',
        'vals': [-4.0679, -7.4837, -1.2819, 0.1618, -6.6281],
        'stds': [ 2.6862,  1.4365,  1.1687, 0.0078,  5.9706],
        'ylim': (-13.5, 1.8),
        'annotation': ('Best in Case B: HGT', 3, 'bottom'),
        'best_label': 3,  # HGT index
    },
    'C': {
        'title': 'Case C: U.S. County Poverty',
        'subtitle': '(Regional separation, >500 km)',
        'vals': [-0.1655, -0.1429, 0.6412, 0.4585, 0.1478],
        'stds': [ 0.0166,  0.2553,  0.0236, 0.0766, 0.2034],
        'ylim': (-0.65, 0.90),
        'annotation': ('Best in Case C: CrossAttn', 2, 'bottom'),
        'best_label': 2,  # CrossAttn index
    },
    'D': {
        'title': 'Case D: EuroSAT  [Exploratory]',
        'subtitle': '(Border split, classification)',
        'vals': [0.878, 0.879, 0.871, None, None],
        'stds': [None, None, None, None, None],
        'ylim': (0.75, 1.02),
        'annotation': None,
        'best_label': None,
    },
}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.linewidth': 1.0})

fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.32})

ax_map = {'A': axes[0,0], 'B': axes[0,1], 'C': axes[1,0], 'D': axes[1,1]}

for case_key, cdata in CASES.items():
    ax = ax_map[case_key]
    vals = cdata['vals']
    stds = cdata['stds']
    x = np.arange(len(MODEL_LABELS))

    # bars (skip None)
    for i, (v, s, c) in enumerate(zip(vals, stds, BAR_COLORS)):
        if v is None:
            ax.bar(i, 0, color='#ccc', width=0.6, edgecolor='none')
            y0, y1 = cdata['ylim']
            ax.text(i, y0 + 0.02 * (y1 - y0), 'N/A',
                    ha='center', va='bottom', fontsize=7.5, color='#888')
            continue
        bar = ax.bar(i, v, color=c, width=0.6, edgecolor='none', linewidth=0, zorder=3,
                     alpha=0.85)
        # error bar
        if s is not None:
            ax.errorbar(i, v, yerr=s, fmt='none', color='#333', capsize=3, linewidth=1, zorder=7)
        # value label
        off = 0.015 * (cdata['ylim'][1] - cdata['ylim'][0])
        va = 'bottom' if v >= 0 else 'top'
        ax.text(i, v + (off if v >= 0 else -off), f'{v:.3f}',
                ha='center', va=va, fontsize=7.5, color='#222', zorder=8)


    ax.axhline(0, color='#555', linewidth=0.9, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylabel('R²' if case_key != 'D' else 'Accuracy', fontsize=9)
    ax.set_ylim(cdata['ylim'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{cdata['title']}\n{cdata['subtitle']}", fontsize=9.5, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(BAR_COLORS, ['EKG', 'Linear Fusion', 'CrossAttn', 'HGT', 'FastGTN'])]
fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
fig.suptitle('Mean performance with across-seed variability across four cases under strict OOD',
             fontsize=14, fontweight='bold', y=0.995)
fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.11, hspace=0.45, wspace=0.32)

fig.savefig(OUT + '/Figure4_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '/Figure4_model_comparison.pdf', bbox_inches='tight', facecolor='white')
print("Figure 3 saved.")
plt.close(fig)
