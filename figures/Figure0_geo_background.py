# -*- coding: utf-8 -*-
"""
Figure 0: Geographic data source overview
Main map (numbered regions) + 2 insets (SE Tibet, U.S. datasets)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import geopandas as gpd
import geodatasets
from shapely.geometry import box, Polygon
import numpy as np

OUT  = "F:/pythoncode/大模型项目/分段架构_DEM/补充数据/实验报告/figures"
land = gpd.read_file(geodatasets.get_path('naturalearth land'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

C_BLUE  = '#4A6FA5'
C_RED   = '#C04D36'
C_GOLD  = '#A68A5A'
C_TRAIN = '#4A6FA5'
C_BUF   = '#BBBBBB'
C_TEST  = '#E07050'

fig = plt.figure(figsize=(14, 8))

# ── Main map (top 70%) ────────────────────────────────────────────────────────
ax = fig.add_axes([0.0, 0.30, 1.0, 0.68])
land.plot(ax=ax, color='#EBEBEB', edgecolor='#C8C8C8', linewidth=0.35)

# ① U.S. — clip land to contiguous US bbox for cleaner outline
us_gdf = gpd.GeoDataFrame(geometry=[box(-125, 24, -66, 50)], crs='EPSG:4326')
us_land = gpd.clip(land, us_gdf)
us_land.plot(ax=ax, color=C_RED, alpha=0.30, zorder=3, edgecolor=C_RED, linewidth=0.5)

# ② SE Tibet — clip to land
tibet_gdf = gpd.GeoDataFrame(geometry=[box(94, 28, 97, 30)], crs='EPSG:4326')
tibet_land = gpd.clip(land, tibet_gdf)
tibet_land.plot(ax=ax, color=C_BLUE, alpha=0.75, zorder=4, edgecolor=C_BLUE, linewidth=1.0)

# ③ Europe — clip to land, weak
eu_gdf = gpd.GeoDataFrame(geometry=[box(-10, 35, 30, 65)], crs='EPSG:4326')
eu_land = gpd.clip(land, eu_gdf)
eu_land.plot(ax=ax, color=C_GOLD, alpha=0.28, zorder=2, edgecolor=C_GOLD, linewidth=0.5)

ax.set_xlim(-170, 180); ax.set_ylim(-58, 82)
ax.axis('off')

# Numbered labels
ax.text(-95, 52, '①  Cases B–C\nU.S.-based datasets',
        fontsize=9.5, color='#6a1010', fontweight='bold', zorder=5, ha='center')
ax.text(108, 36, '②  Case A: Forest Biomass\nSE Tibet',
        fontsize=9.5, color='#1a3a5c', fontweight='bold', zorder=5, ha='left')
ax.text(10, 68, '③  Case D: EuroSAT\n(exploratory)',
        fontsize=8.5, color='#7a6030', style='italic', zorder=5, ha='center')

# Short arrows from labels to regions
ax.annotate('', xy=(-96, 37), xytext=(-95, 50),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))
ax.annotate('', xy=(97, 29), xytext=(108, 36),
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2))

# Zoom indicator boxes on main map
for (x0,y0,w,h), color in [
    ((-125,24,59,26), C_RED),   # US box
    ((94,28,3,2),     C_BLUE),  # Tibet box
]:
    ax.add_patch(mpatches.Rectangle((x0,y0), w, h,
        fill=False, edgecolor=color, linewidth=1.5, linestyle='--', zorder=6))

# ── Inset ① : U.S. datasets ──────────────────────────────────────────────────
ax_u = fig.add_axes([0.52, 0.02, 0.23, 0.30])
land.plot(ax=ax_u, color='#EBEBEB', edgecolor='#C8C8C8', linewidth=0.5)

# Layer 1: contiguous US (light)
us_gdf.plot(ax=ax_u, color=C_RED, alpha=0.15, zorder=2)

# Layer 2: California (stronger)
ca_gdf = gpd.GeoDataFrame(geometry=[box(-124.35, 32.54, -114.31, 41.95)], crs='EPSG:4326')
ca_gdf.plot(ax=ax_u, color=C_RED, alpha=0.55, zorder=3)

# Case labels — two clear tiers
ax_u.text(-119.3, 43.5, 'Case B: California Housing',
          fontsize=7, color='#6a1010', fontweight='bold', ha='center')
ax_u.text(-96, 51.0, 'Case C: County Poverty  (contiguous U.S.)',
          fontsize=6.5, color='#7a1a10', ha='center', style='italic')

# Buffered split schematic — small legend only, no text on bands
ax_u.axvline(-121.0, ymin=0.28, ymax=0.68, color=C_BUF, lw=0.9, ls='--', zorder=5)
ax_u.axvline(-119.5, ymin=0.28, ymax=0.68, color=C_BUF, lw=0.9, ls='--', zorder=5)
split_handles = [
    mpatches.Patch(color=C_TRAIN, alpha=0.5, label='Train'),
    mpatches.Patch(color=C_BUF,   alpha=0.5, label='Buffer'),
    mpatches.Patch(color=C_TEST,  alpha=0.5, label='Test'),
]
ax_u.legend(handles=split_handles, fontsize=5.5, loc='lower left',
            framealpha=0.9, handlelength=1.0, title='buffered split\n(schematic)',
            title_fontsize=5)

ax_u.set_xlim(-128, -63); ax_u.set_ylim(22.5, 52)
ax_u.set_title('① U.S. datasets', fontsize=8, fontweight='bold', color='#6a1010', pad=3)
ax_u.tick_params(labelsize=5.5)
ax_u.set_xlabel('Longitude', fontsize=6); ax_u.set_ylabel('Latitude', fontsize=6)

# ── Inset ② : SE Tibet ───────────────────────────────────────────────────────
ax_t = fig.add_axes([0.77, 0.02, 0.22, 0.30])

# Land base (zoomed to SE Tibet region)
land.plot(ax=ax_t, color='#EBEBEB', edgecolor='#C8C8C8', linewidth=0.5)

# Three bands clipped to research area polygon
tibet_poly = Polygon([(94.0,28.1),(94.6,27.9),(96.1,28.0),(97.0,28.3),
                       (97.0,29.9),(96.3,30.2),(94.9,30.0),(94.0,29.4)])
for xmin, xmax, color in [(94.0,95.3,C_TRAIN),(95.3,95.9,C_BUF),(95.9,97.0,C_TEST)]:
    band = box(xmin, 27.3, xmax, 30.8)
    clipped = tibet_poly.intersection(band)
    if not clipped.is_empty:
        gpd.GeoDataFrame(geometry=[clipped], crs='EPSG:4326').plot(
            ax=ax_t, color=color, alpha=0.30, zorder=1)

# Research area outline
gpd.GeoDataFrame(geometry=[tibet_poly], crs='EPSG:4326').plot(
    ax=ax_t, facecolor='none', edgecolor='#333', linewidth=1.4, zorder=3)

# Band labels (outside polygon, below)
ax_t.text(94.65, 27.55, 'Train', fontsize=6.5, color=C_TRAIN, ha='center', fontweight='bold')
ax_t.text(95.60, 27.55, 'Buffer', fontsize=6.5, color='#666', ha='center')
ax_t.text(96.45, 27.55, 'Test', fontsize=6.5, color=C_TEST, ha='center', fontweight='bold')
ax_t.text(95.5, 30.55, 'buffered split (schematic)',
          fontsize=5.5, color='#888', ha='center', style='italic')

# Legend
legend_handles = [
    mpatches.Patch(color=C_TRAIN, alpha=0.5, label='Train'),
    mpatches.Patch(color=C_BUF,   alpha=0.5, label='Buffer'),
    mpatches.Patch(color=C_TEST,  alpha=0.5, label='Test'),
]
ax_t.legend(handles=legend_handles, fontsize=5.5, loc='lower right',
            framealpha=0.9, handlelength=1.0)

ax_t.set_xlim(93.5, 97.5); ax_t.set_ylim(27.3, 30.8)
ax_t.set_title('② Case A: SE Tibet', fontsize=8, fontweight='bold', color='#1a3a5c', pad=3)
ax_t.tick_params(labelsize=5.5)
ax_t.set_xlabel('Longitude', fontsize=6); ax_t.set_ylabel('Latitude', fontsize=6)
ax_t.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

plt.savefig(OUT + '/Figure0_geo_background.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT + '/Figure0_geo_background.pdf', bbox_inches='tight', facecolor='white')
print("Figure 0 saved.")
plt.close()
