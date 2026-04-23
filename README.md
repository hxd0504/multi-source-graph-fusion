# Multi-Source Graph Fusion for Remote Sensing Regression

A dual-graph neural network framework that fuses spatial topology and ecological feature graphs for above-ground biomass (AGB) prediction from multi-source remote sensing data.

## Overview

This project proposes a **segmented dual-graph fusion architecture** combining:
- **TKG** (Topological Knowledge Graph): spatial KNN graph processed by GAT
- **EKG** (Ecological Knowledge Graph): cosine-similarity feature graph processed by GCN
- **Segmented training**: threshold-based regime splitting to handle heterogeneous biomass distributions

The framework is validated on EuroSAT RGB and real-world multi-year remote sensing data (NDVI + ESA AGB + DEM) over the Tibetan Plateau.

## Project Structure

```
├── models/
│   ├── gnn_models.py      # TKGOnly, EKGOnly, FusionModel definitions
│   ├── trainer.py         # Segmented & standard training routines
│   └── data_utils.py      # Data loading, TKG/EKG graph construction
├── experiments/
│   ├── exp1_segmentation_ablation.py    # Segmented vs. non-segmented
│   ├── exp2_tkg_ekg_ablation.py         # TKG-only / EKG-only / Fusion
│   ├── exp3_boundary_robustness.py      # Boundary region robustness
│   ├── exp4_spatial_generalization.py   # OOD spatial generalization
│   └── exp5_graph_structure_analysis.py # Graph topology analysis
├── figures/               # Paper figure generation scripts
├── data/                  # Data directory (see data/README.md)
└── run_all_experiments.py # One-click experiment runner
```

## Key Technical Contributions

1. **Dual-graph fusion**: TKG captures geographic proximity; EKG captures semantic similarity between land patches — fused via learned attention weights.
2. **Segmented modeling**: A pre-trained RF initializer assigns each sample to a biomass regime before graph training, avoiding label leakage at test time.
3. **Strict OOD evaluation**: Spatial buffer-based train/test split ensures no geographic overlap, testing true generalization.

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `torch-geometric`, `scikit-learn`, `numpy`, `matplotlib`, `rasterio`

## Usage

```bash
# Run all experiments
python run_all_experiments.py

# Run individual experiment
python experiments/exp1_segmentation_ablation.py
```

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| TKG-only | — | — | — |
| EKG-only | — | — | — |
| **Fusion (ours)** | **—** | **—** | **—** |

等待论文展出后公布具体数值

## Paper

Submitted to *Engineering Applications of Artificial Intelligence* (EAAI).
