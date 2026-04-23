# -*- coding: utf-8 -*-
"""
Figure 7: Attention Entropy Distributions across Representative Cases
Data: per-node normalized attention entropy (train-fit/test-transform)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT      = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"
RES      = "F:/pythoncode/大模型项目/分段架构_DEM/补充实验/results"

biomass  = np.load(f'{RES}/forest_biomass_hattn_entropy.npy')
housing  = np.load(f'{RES}/california_housing_hattn_entropy.npy')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

CASES = [
    (housing, 'Case B: California Housing', '#A8C5DA'),
    (biomass, 'Case A: Forest Biomass',    '#4A6FA5'),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

for ax, (data, title, color) in zip(axes, CASES):
    ax.hist(data, bins=25, color=color, edgecolor='white', linewidth=0.4,
            density=True, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized attention entropy', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f'{OUT}/Figure7_attention_entropy.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/Figure7_attention_entropy.pdf', bbox_inches='tight', facecolor='white')
print("Figure 7 saved.")
plt.close(fig)
