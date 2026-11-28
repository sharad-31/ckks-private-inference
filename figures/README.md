# Figures & Visual Generation

All report and paper figures (300 DPI PNGs) are generated reproducibly from the experimental outputs in `../tabular_mimic/reports/` and `../image_malaria/reports/` by running:

```bash
python generate_figures.py
```

| File | Description | Data Source |
|---|---|---|
| `fig_1.png` | Client–server CKKS workflow (LR & CNN) | Architecture schematic |
| `fig_2.png` | `im2col` per-patient vs. feature-major SIMD batching layout | Encoding schematic |
| `fig_3.png` | Tabular test AUROC & accuracy vs. always-negative baseline | `tabular_mimic/reports/train_*.json`, `ckks_*.json` |
| `fig_4.png` | Test-split class flips across all tasks & presets | `tabular_mimic/reports/ckks_*.json`, `image_malaria/reports/ckks_*.json` |
| `fig_5.png` | Distribution of `|encrypted − plaintext logit|` | `tabular_mimic/reports/predictions_*.csv`, `image_malaria/reports/predictions_*.csv` |
| `fig_6.png` | Image test metrics & confusion matrices (LR vs. CNN) | `image_malaria/reports/predictions_*.csv`, `image_malaria/models/*.json` |
| `fig_7.png` | Wall-clock cost (tabular per-patient & image per-patient vs. SIMD batch) | `tabular_mimic/reports/ckks_*.json`, `image_malaria/reports/ckks_*.json`, `batch_*.json` |
| `fig_8.png` | Batched encrypted threshold (saturated outputs & class flips) | `image_malaria/reports/batch_*_decision*.json` |
