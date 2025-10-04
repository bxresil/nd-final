#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze ND transmission (A segment, white screen).

Assumptions:
- Files are named with ND stop inside the filename, e.g. ...ND0..., ...ND1..., ...ND5...
- Brightness per image ~ average pixel value.

Outputs:
- log2T_vs_n.png : log2(transmission) vs ND stop
- summary.json   : slope of fit and raw points

Usage:
python analyze_nd.py --data_dir ~/Desktop/ND_A_raw --out_dir ~/Desktop/ND_A_out
"""

import os, re, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ASCII-only labels and a common font to avoid missing glyphs
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ND_RE = re.compile(r"ND(\d+)", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def list_images(d):
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith(exts)
    )

def get_nd_from_name(path):
    m = ND_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def image_mean(path):
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float64)
    return float(np.mean(arr))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_nd = {}
    for p in list_images(args.data_dir):
        nd = get_nd_from_name(p)
        if nd is None:
            continue
        by_nd.setdefault(nd, []).append(image_mean(p))

    if not by_nd:
        raise RuntimeError("No images containing 'ND*' found in filenames.")

    nds_sorted = sorted(by_nd.keys())
    means = np.array([np.mean(by_nd[n]) for n in nds_sorted], dtype=np.float64)

    # Transmission relative to ND0
    if 0 not in nds_sorted:
        raise RuntimeError("ND0 is required as the baseline.")
    T = means / (means[nds_sorted.index(0)] + 1e-12)

    x = np.array(nds_sorted, dtype=np.float64)
    y = np.log2(T)

    # Linear fit: y = a + b*x
    b, a = np.polyfit(x, y, 1)

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(x, y, "o", label="log2(T) points")
    ax.plot(x, a + b*x, "-", label=f"fit: slope={b:.3f}")
    ax.plot(x, -1.0*x, "--", label="ideal slope -1")
    ax.set_title("ND attenuation")
    ax.set_xlabel("ND n")
    ax.set_ylabel("log2(T)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nds_sorted)
    ax.legend()
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "log2T_vs_n.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"nd": nds_sorted, "log2T": y.tolist(), "slope": float(b)}, f, indent=2)

    print("Done.")
    print(f"Plot: {out_png}")

if __name__ == "__main__":
    main()
