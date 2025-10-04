#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make plots from ColorChecker CSV (patch_means.csv), grouped by temperature and ND stop.

Inputs
------
- CSV from nd_colorchecker_batch.py with columns:
  filename, R_0_0, G_0_0, B_0_0, ..., R_3_5, G_3_5, B_3_5  (24 patches x 3)
- Filenames must contain patterns like: C_3000K_ND3_color.TIF (temp and ND)
- Optionally normalize overall brightness by the average G channel per image.

Outputs
-------
- 4 plots in out folder:
  * 3000K_channel_shifts_vs_n.png   (dRG%, dBG%)
  * 3000K_deLab_vs_n.png            (dEab mean, dEab max vs ND0)
  * 6000K_channel_shifts_vs_n.png
  * 6000K_deLab_vs_n.png
- color_shifts_summary.csv: numeric table used for plots

Usage
-----
python cc_fromcsv_make_plots.py \
  --csv ~/Desktop/ND_B_out_24patch/patch_means.csv \
  --out ~/Desktop/ND_B_out_24patch \
  --norm G
"""

import os, re, csv, argparse
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# Use a common ASCII-safe font and sane defaults
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ND_KEY_RE = re.compile(r"ND(\d+)", re.IGNORECASE)
TEMP_RE   = re.compile(r"C_(\d{3,5})K", re.IGNORECASE)

PATCH_IDX = [(i, j) for i in range(4) for j in range(6)]  # 4x6 = 24

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to patch_means.csv")
    ap.add_argument("--out", required=True, help="Folder to write plots and summary")
    ap.add_argument("--norm", default=None, choices=[None, "G", "g"], help="Optional brightness normalization by avg G")
    return ap.parse_args()

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = rows[1:]
    return header, data

def extract_temp_and_nd(filename):
    # temperature
    tmatch = TEMP_RE.search(filename)
    temp = int(tmatch.group(1)) if tmatch else None
    # ND stop
    nmatch = ND_KEY_RE.search(filename)
    nd = int(nmatch.group(1)) if nmatch else None
    return temp, nd

def row_to_rgb_arrays(header, row):
    # returns np.array of shape (24,) for R, G, B
    def find_col(prefix):
        cols = []
        for i, j in PATCH_IDX:
            cols.append(header.index(f"{prefix}_{i}_{j}"))
        return cols
    r_cols = find_col("R")
    g_cols = find_col("G")
    b_cols = find_col("B")
    R = np.array([float(row[c]) for c in r_cols], dtype=np.float64)
    G = np.array([float(row[c]) for c in g_cols], dtype=np.float64)
    B = np.array([float(row[c]) for c in b_cols], dtype=np.float64)
    return R, G, B

# ---- sRGB D65 to Lab helpers (for relative comparisons) ----
# Based on standard formulas. Inputs are in [0,1].
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def rgb_to_xyz(R, G, B):
    # sRGB to XYZ (D65)
    r = srgb_to_linear(R); g = srgb_to_linear(G); b = srgb_to_linear(B)
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return X, Y, Z

def f_lab(t):
    epsilon = 216/24389
    kappa   = 24389/27
    return np.where(t > epsilon, np.cbrt(t), (kappa*t + 16)/116)

def xyz_to_lab(X, Y, Z):
    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx, fy, fz = f_lab(X/Xn), f_lab(Y/Yn), f_lab(Z/Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return L, a, b

def rgb_to_lab(R, G, B):
    X, Y, Z = rgb_to_xyz(R, G, B)
    return xyz_to_lab(X, Y, Z)

def deltaE76(L1, a1, b1, L2, a2, b2):
    return np.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)

def compute_group_metrics(rows, header, norm_by_g=False):
    """
    Returns a nested dict:
    metrics[temp]['nd'] = sorted list of nd stops
    metrics[temp]['rg'] = list of RG values (R/G) per nd
    metrics[temp]['bg'] = list of BG values (B/G) per nd
    metrics[temp]['de_mean'] = list of mean dEab vs ND0 per nd
    metrics[temp]['de_max']  = list of max dEab vs ND0 per nd
    """
    buckets = defaultdict(list)
    for row in rows:
        fname = row[0]
        temp, nd = extract_temp_and_nd(fname)
        if temp is None or nd is None:
            continue
        R, G, B = row_to_rgb_arrays(header, row)
        if norm_by_g:
            gmean = np.nanmean(G) if np.all(np.isfinite(G)) else 1.0
            if gmean == 0: gmean = 1.0
            R, G, B = R/gmean, G/gmean, B/gmean

        # per-image simple RG, BG using means across patches
        RG = float(np.nanmean(R) / (np.nanmean(G) + 1e-12))
        BG = float(np.nanmean(B) / (np.nanmean(G) + 1e-12))

        # prepare Lab arrays per patch in [0,1] (clip large values robustly)
        # scale by 99th percentile to map into [0,1] reasonably
        scale = np.percentile(np.r_[R,G,B], 99)
        if scale <= 0: scale = 1.0
        Rs = np.clip(R/scale, 0, 1)
        Gs = np.clip(G/scale, 0, 1)
        Bs = np.clip(B/scale, 0, 1)
        L, A, Bb = rgb_to_lab(Rs, Gs, Bs)

        buckets[(temp, nd)].append({
            "fname": fname, "RG": RG, "BG": BG,
            "L": L, "A": A, "B": Bb
        })

    # Collapse multiple entries per (temp, nd) by averaging
    collapsed = defaultdict(dict)
    temps = set()
    for (temp, nd), lst in buckets.items():
        temps.add(temp)
        RG = np.mean([d["RG"] for d in lst])
        BG = np.mean([d["BG"] for d in lst])
        L  = np.mean([d["L"]  for d in lst], axis=0)
        A  = np.mean([d["A"]  for d in lst], axis=0)
        Bb = np.mean([d["B"]  for d in lst], axis=0)
        collapsed[temp][nd] = {"RG": RG, "BG": BG, "L": L, "A": A, "B": Bb}

    # Build metrics vs ND with deltas to ND0
    metrics = {}
    for temp in sorted(list(temps)):
        nds = sorted(collapsed[temp].keys())
        # baselines (ND0 required)
        if 0 not in collapsed[temp]:
            # skip temp if no ND0
            continue
        base = collapsed[temp][0]
        RG0, BG0 = base["RG"], base["BG"]
        L0, A0, B0 = base["L"], base["A"], base["B"]

        arr_nd, arr_rg, arr_bg, de_mean, de_max = [], [], [], [], []
        for nd in nds:
            cur = collapsed[temp][nd]
            RG, BG = cur["RG"], cur["BG"]
            dRG = (RG / (RG0 + 1e-12) - 1.0) * 100.0
            dBG = (BG / (BG0 + 1e-12) - 1.0) * 100.0
            dE  = deltaE76(cur["L"], cur["A"], cur["B"], L0, A0, B0)
            arr_nd.append(nd); arr_rg.append(dRG); arr_bg.append(dBG)
            de_mean.append(float(np.mean(dE)))
            de_max.append(float(np.max(dE)))

        metrics[temp] = {
            "nd": arr_nd,
            "dRG": arr_rg,
            "dBG": arr_bg,
            "dE_mean": de_mean,
            "dE_max": de_max
        }
    return metrics

def save_summary_table(metrics, out_csv):
    # rows: temp, nd, dRG%, dBG%, dE_mean, dE_max
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tempK", "nd", "dRG_percent", "dBG_percent", "dEab_mean", "dEab_max"])
        for temp in sorted(metrics.keys()):
            m = metrics[temp]
            for nd, drg, dbg, de1, de2 in zip(m["nd"], m["dRG"], m["dBG"], m["dE_mean"], m["dE_max"]):
                w.writerow([temp, nd, f"{drg:.6f}", f"{dbg:.6f}", f"{de1:.6f}", f"{de2:.6f}"])

def plot_channel_shifts(temp, nd, dRG, dBG, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(nd, dRG, "o-", label="dRG %")
    ax.plot(nd, dBG, "s--", label="dBG %")
    ax.set_title(f"Channel ratio shifts vs ND — {temp}K (norm=G)")
    ax.set_xlabel("ND n")
    ax.set_ylabel("Ratio change (%)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(nd))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_de(temp, nd, dE_mean, dE_max, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(nd, dE_mean, "o-", label="dEab mean")
    ax.plot(nd, dE_max,  "s--", label="dEab max")
    ax.set_title(f"Color difference vs ND — {temp}K (norm=G)")
    ax.set_xlabel("ND n")
    ax.set_ylabel("dEab (vs ND0)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(nd))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    header, rows = load_csv(args.csv)
    norm_by_g = (args.norm is not None and args.norm.upper() == "G")

    metrics = compute_group_metrics(rows, header, norm_by_g=norm_by_g)
    out_csv = os.path.join(args.out, "color_shifts_summary.csv")
    save_summary_table(metrics, out_csv)

    for temp, m in metrics.items():
        ch_path = os.path.join(args.out, f"{temp}K_channel_shifts_vs_n.png")
        de_path = os.path.join(args.out, f"{temp}K_deLab_vs_n.png")
        plot_channel_shifts(temp, m["nd"], m["dRG"], m["dBG"], ch_path)
        plot_de(temp, m["nd"], m["dE_mean"], m["dE_max"], de_path)

    print("Done.")
    print(f"Wrote: {out_csv}")
    print(f"Plots in: {args.out}")

if __name__ == "__main__":
    main()
