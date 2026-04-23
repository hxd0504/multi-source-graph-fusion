# -*- coding: utf-8 -*-
"""
Figure 8b: Conceptual ranking shifts across strict OOD cases
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

MODELS = ['Single-source\nbaseline', 'Linear\nfusion', 'Cross-attention\nfusion', 'Relation-aware\n(HGT/FastGTN)']
# y positions per case: higher = better rank
RANKS = {
    'Case A\nForest Biomass':      [0.55, 0.60, 0.50, 0.45],   # clustered, no stable winner
    'Case B\nCalifornia Housing':  [0.30, 0.35, 0.55, 0.75],   # HGT leads
    'Case C\nU.S. County Poverty': [0.25, 0.30, 0.80, 0.55],   # CrossAttn leads
}
COLORS = ['#888888', '#C04D36', '#7C2E36', '#4A6FA5']
NOTES  = {
    'Case A\nForest Biomass':      'No stable winner\n(boundary case)',
    'Case B\nCalifornia Housing':  'HGT leads\nin this case',
    'Case C\nU.S. County Poverty': 'Cross-Attention\nremains strongest',
}
CASE_COLORS = ['#A68A5A', '#4A6FA5', '#7C2E36']

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

for ax, (case, ranks), cc in zip(axes, RANKS.items(), CASE_COLORS):
    for i, (model, y, mc) in enumerate(zip(MODELS, ranks, COLORS)):
        ax.scatter(0.5, y, s=120, color=mc, zorder=3, edgecolors='white', linewidths=1)
        ax.text(0.52, y, model, va='center', fontsize=8.5, color='#333')

    # light connector line
    ax.plot([0.5]*len(ranks), ranks, color='#DDDDDD', linewidth=1, zorder=1)

    ax.set_title(case, fontsize=10, fontweight='bold', color=cc)
    ax.text(0.5, 0.08, NOTES[case], ha='center', va='bottom', fontsize=8,
            color='#666', style='italic', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.axis('off')

# shared y-label via fig text
fig.text(0.01, 0.5, 'Relative ranking (higher = better)', va='center',
         rotation='vertical', fontsize=10, color='#444')

fig.text(0.5, 0.01,
         'Relative ranking patterns vary across cases under strict OOD; no single model family dominates all settings.',
         ha='center', fontsize=8, color='#888')

# legend
handles = [mpatches.Patch(color=c, label=m.replace('\n', ' ')) for m, c in zip(MODELS, COLORS)]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

plt.tight_layout(rect=[0.04, 0.06, 1, 0.97])
fig.savefig(OUT + '/Figure8b_concept_ranking.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '/Figure8b_concept_ranking.pdf', bbox_inches='tight', facecolor='white')
print("Figure 8b saved.")
plt.close(fig)
