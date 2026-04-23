# Data

Raw data files are not included in this repository due to size constraints.

## Required Files

Place the following files in this directory before running experiments:

- `aligned_features_multiyear.npy` — 12-dim feature vectors (4000 nodes × 12)
- `aligned_labels_multiyear.npy` — AGB labels in Mg/ha
- `aligned_metadata_multiyear.pkl` — coordinates and year metadata
- `EuroSAT_RGB/` — EuroSAT RGB dataset (download from [EuroSAT](https://github.com/phelber/EuroSAT))

## Data Sources

| Source | Resolution | Years | Variables |
|--------|-----------|-------|-----------|
| ESA AGB | 100m | 2010, 2015–2021 | Above-ground biomass |
| MODIS NDVI | 30m | 2010, 2015–2021 | Vegetation index |
| SRTM DEM | 30m | static | Elevation, slope, aspect, curvature |

Study area: 95–96°E, 29–30°N (southeastern Tibet)
