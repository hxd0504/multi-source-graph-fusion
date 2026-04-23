# -*- coding: utf-8 -*-
"""Fix Figure1: OOD Split Visualization - Block + Buffer instead of radial"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# S1: Random Split (IID)
ax1 = axes[0]
np.random.seed(42)
n = 100
x1 = np.random.rand(n)
y1 = np.random.rand(n)
train_mask = np.random.rand(n) > 0.3
ax1.scatter(x1[train_mask], y1[train_mask], c='#2196F3', s=30, alpha=0.7, label='Train')
ax1.scatter(x1[~train_mask], y1[~train_mask], c='#FF9800', s=30, alpha=0.7, label='Test')
ax1.set_title('S1: Random Split (IID)', fontweight='bold', fontsize=11)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend(loc='upper right', fontsize=8)

# S2: Spatial Block Split (not CV)
ax2 = axes[1]
ax2.add_patch(mpatches.Rectangle((0, 0), 0.6, 1, fc='#2196F3', alpha=0.3, label='Train'))
ax2.add_patch(mpatches.Rectangle((0.6, 0), 0.4, 1, fc='#FF9800', alpha=0.3, label='Test'))
ax2.axvline(0.6, color='#333', linewidth=2, linestyle='--')
ax2.text(0.3, 0.5, 'Train Block', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(0.8, 0.5, 'Test Block', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.set_title('S2: Spatial Block Split', fontweight='bold', fontsize=11)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticks([])
ax2.set_yticks([])

# S3: Strict OOD with Buffer (Block + Buffer)
ax3 = axes[2]
ax3.add_patch(mpatches.Rectangle((0, 0), 0.45, 1, fc='#2196F3', alpha=0.3, label='Train'))
ax3.add_patch(mpatches.Rectangle((0.45, 0), 0.15, 1, fc='#9E9E9E', alpha=0.4, label='Buffer (excluded)'))
ax3.add_patch(mpatches.Rectangle((0.6, 0), 0.4, 1, fc='#FF9800', alpha=0.3, label='Test'))
ax3.text(0.225, 0.5, 'Train\nBlock', ha='center', va='center', fontsize=9, fontweight='bold')
ax3.text(0.525, 0.5, 'Buffer', ha='center', va='center', fontsize=8, color='#555')
ax3.text(0.8, 0.5, 'Test\nBlock', ha='center', va='center', fontsize=9, fontweight='bold')
ax3.set_title('S3: Strict OOD with Buffer', fontweight='bold', fontsize=11)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks([])
ax3.set_yticks([])

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

handles = [mpatches.Patch(color='#2196F3', alpha=0.5, label='Train'),
           mpatches.Patch(color='#FF9800', alpha=0.5, label='Test'),
           mpatches.Patch(color='#9E9E9E', alpha=0.5, label='Buffer (excluded)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig('F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures/Figure1_ood_split_visualization.png',
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/paper_submission_final/Figure1_ood_split_visualization.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 fixed: S2 changed to 'Block Split', S3 changed to block+buffer layout.")
plt.close(fig)
