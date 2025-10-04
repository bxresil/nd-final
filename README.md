# ND Filter Experiment — Final Package

## Package layout

ND_FINAL/
├─ code/
│ ├─ nd_colorchecker_batch.py # interactive rectification + 4×6 ROI + CSV
│ ├─ cc_fromcsv_make_plots.py # build plots from CSV
│ └─ analyze_nd.py # ND attenuation (log2(T) vs ND n)
├─ data/
│ ├─ patch_means.csv # 24-patch RGB means per image
│ └─ color_shifts_summary.csv # summary used for plots
├─ figs/
│ ├─ 3000K_channel_shifts_vs_n.png
│ ├─ 3000K_deLab_vs_n.png
│ ├─ 6000K_channel_shifts_vs_n.png
│ ├─ 6000K_deLab_vs_n.png
│ └─ ND_attenuation_screen.png # segment A (screen-light series)
└─ roi/
├─ *_grid_overlay.png # ROI proof overlays on warped images
└─ *_warped.png # rectified (top-down) ColorChecker images

> Note: `analyze_colorchecker_deltaE_any.py` is **not used** in this deliverable.

---

## Requirements
- Python 3
- Packages: `numpy`, `matplotlib`, `opencv-python`

Install (if needed):
```bash
pip install -U numpy matplotlib opencv-python

Reproduce segment B (24-patch workflow)
1) Generate warped images + ROI overlays + CSV (optional re-run)

By default, nd_colorchecker_batch.py reads TIFFs from ~/Desktop/ND_B_raw and writes to ~/Desktop/ND_B_out_24patch.
python code/nd_colorchecker_batch.py

This produces:

*_warped.png, *_grid_overlay.png

patch_means.csv (24 patch means per image)

If you want these warped/overlay images inside the package:
mkdir -p ~/Desktop/ND_FINAL/roi
cp ~/Desktop/ND_B_out_24patch/*_warped.png       ~/Desktop/ND_FINAL/roi/
cp ~/Desktop/ND_B_out_24patch/*_grid_overlay.png ~/Desktop/ND_FINAL/roi/

2) Build plots from the CSV included in data/
python code/cc_fromcsv_make_plots.py \
  --csv data/patch_means.csv \
  --out figs \
  --norm G

--norm G uses the green channel to normalize per-image brightness before computing channel ratios and ΔE.

Outputs go to figs/.

Reproduce segment A (ND attenuation)

If you want to regenerate the attenuation plot for the “screen-light” series:
python code/analyze_nd.py \
  --data_dir <path_to_ND_A_raw> \
  --out_dir figs

This writes ND_attenuation_screen.png to figs/.

Notes & assumptions

ROI placement proof: use files in roi/ (*_grid_overlay.png) to visually verify that each 4×6 cell’s inner safety margin only samples inside the patch.

Channel-ratio plots and ΔE plots depend on illumination spectrum and geometry. --norm G removes overall brightness drift but preserves spectral bias trends through ND steps.

