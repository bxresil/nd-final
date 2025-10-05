# ND Filter Experiment — Final Package

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Summary
This repo studies **variable ND filters** in the RAW domain.  
We (i) measure transmission vs. nominal stops,  
(ii) quantify color neutrality (ΔE*ab and channel ratios) under **3000K / 6000K**,  
and (iii) provide a fully reproducible pipeline.

**Key findings**
- Transmission closely follows **T ≈ 2⁻ⁿ** with small deviations at higher densities.
- Color neutrality drifts with density; **warm vs. cool** illuminants bias differently.
- All figures and CSVs are reproducible via scripts in `code/`.

## Results (quick view)
<p float="left">
  <img src="figs/ND_attenuation_screen.png" width="44%" />
  <img src="figs/3000K_deLab_vs_n.png" width="27%" />
  <img src="figs/6000K_deLab_vs_n.png" width="27%" />
</p>

---

## Package layout
- `code/` — Python scripts
- `data/` — CSV outputs
- `figs/` — Figures
- `roi/` — ROI overlays

---

## Requirements
- Python 3
- Install with:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

[200~EOF
