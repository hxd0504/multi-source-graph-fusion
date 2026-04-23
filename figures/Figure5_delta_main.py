# -*- coding: utf-8 -*-
"""
Figure 5: Cost–benefit small multiples (3 panels, one per case)
Each panel: 4 models (LinearFusion, CrossAttn, HGT, FastGTN)
X: relative training time  Y: ΔR² vs. best single-source (TKG/EKG)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# ── Data ─────────────────────────────────────────────────────────────────────
BEST_SINGLE = {'A': -0.0086, 'B': -4.0679, 'C': -0.1655}

MODEL_R2 = {
    'LinearFusion': {'A':  0.0072, 'B': -7.4837, 'C': -0.1429},
    'CrossAttn':    {'A': -0.2329, 'B': -1.2819, 'C':  0.6412},
    'HGT':          {'A':  0.0503, 'B':  0.1618, 'C':  0.4585},
    'FastGTN':      {'A':  0.1061, 'B': -6.6281, 'C':  0.1478},
}

# estimated relative training time (single-source = 1×)
REL_TIME = {'LinearFusion': 1.3, 'CrossAttn': 2.5, 'HGT': 3.5, 'FastGTN': 4.0}

COLORS = {
    'LinearFusion': '#C04D36',
    'CrossAttn':    '#7C2E36',
    'HGT':          '#A68A5A',
    'FastGTN':      '#F2D69C',
}
LABELS = {
    'LinearFusion': 'Linear Fusion',
    'CrossAttn':    'Cross-Attention',
    'HGT':          'HGT',
    'FastGTN':      'FastGTN',
}
CASE_TITLES = {
    'A': 'Case A: Forest Biomass',
    'B': 'Case B: California Housing',
    'C': 'Case C: County Poverty',
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                         gridspec_kw={'wspace': 0.35})

for ax, case in zip(axes, ['A', 'B', 'C']):
    for model, r2_dict in MODEL_R2.items():
        delta = r2_dict[case] - BEST_SINGLE[case]
        x = REL_TIME[model]
        ax.scatter(x, delta, color=COLORS[model], s=90, zorder=4,
                   linewidths=0, alpha=0.9)

    ax.axhline(0, color='#555', linewidth=1.0, linestyle='--', zorder=2)
    ax.set_title(CASE_TITLES[case], fontsize=10, fontweight='bold')
    ax.set_xlabel('Relative training time (1× = single)', fontsize=9)
    ax.set_ylabel('Performance gain over best single-source baseline (ΔR²)', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.8, 5.2)


# shared legend
handles = [mlines.Line2D([], [], color=COLORS[m], marker='o', linestyle='None',
                         markersize=8, label=LABELS[m]) for m in MODEL_R2]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

fig.savefig(OUT + '/Figure5_delta_main.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '/Figure5_delta_main.pdf', bbox_inches='tight', facecolor='white')
print("Figure 5 saved.")
plt.close(fig)
