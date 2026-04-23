# -*- coding: utf-8 -*-
"""
Figure 6: Test-domain CCA signals across three regression cases under strict OOD
Data source: tableA1_cca_fixed_real_with_caseC.csv (train-fit/test-transform, 2026-04-14)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

cases  = ['Case A\nForest Biomass', 'Case B\nCalifornia Housing', 'Case C\nU.S. County Poverty']
top1   = [0.663, 0.566, 0.237]
mu5    = [0.320, 0.547, 0.168]

x      = np.arange(len(cases))
w      = 0.32
color1 = '#4A6FA5'   # CCA top-1
color2 = '#A8C5DA'   # CCA μ5

fig, ax = plt.subplots(figsize=(7, 4.5))

bars1 = ax.bar(x - w/2, top1, w, color=color1, label=r'$\rho_1$', zorder=3)
bars2 = ax.bar(x + w/2, mu5,  w, color=color2, label=r'$\bar{\rho}_{1:5}$', zorder=3)

for bar, v in zip(bars1, top1):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, color='#222')
for bar, v in zip(bars2, mu5):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, color='#222')

ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=10)
ax.set_ylabel('Test-domain CCA value', fontsize=10)
ax.set_ylim(0, 0.75)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9.5, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT + '/Figure6_cca_testdomain.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '/Figure6_cca_testdomain.pdf', bbox_inches='tight', facecolor='white')
print("Figure 6 saved.")
plt.close(fig)
