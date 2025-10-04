#!/usr/bin/env python3

import os, csv, glob
import numpy as np
import cv2
from pathlib import Path

HOME = str(Path.home())
IN_FOLDER  = os.path.join(HOME, "Desktop", "ND_B_raw")
OUT_FOLDER = os.path.join(HOME, "Desktop", "ND_B_out_24patch")
EXTS       = (".tif", ".TIF")
ROWS, COLS = 4, 6
PATCH_PX   = 220
MARGIN_FR  = 0.06
AUTOSCALE_MAX = 1600
CSV_NAME   = "patch_means.csv"

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder, exts):
    fs=[]
    for e in exts:
        fs += glob.glob(os.path.join(folder, f"*{e}"))
    return sorted(fs)

def resize_for_display(img, max_edge=1600):
    h,w = img.shape[:2]
    if max(h,w) > max_edge:
        s = max_edge/float(max(h,w))
        img2 = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        return img2, s
    return img.copy(), 1.0

class CornerClicker:
    def __init__(self, winname, img_disp, scale):
        self.winname=winname
        self.img_disp=img_disp.copy()
        self.temp=img_disp.copy()
        self.scale=scale
        self.points=[]
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(winname, self._on_mouse)
    def _on_mouse(self, ev,x,y,flags,param):
        if ev==cv2.EVENT_LBUTTONDOWN:
            self.points.append((x,y))
            cv2.circle(self.temp,(x,y),6,(0,255,0),-1)
            cv2.imshow(self.winname,self.temp)
    def run(self):
        print("Click the 4 inner-rectangle corners in order: TL → TR → BR → BL. Enter=confirm, r=reset, q=skip")
        while True:
            cv2.imshow(self.winname, self.temp)
            k=cv2.waitKey(10)&0xFF
            if k in (13,10):
                if len(self.points)==4:
                    break
            elif k==ord('r'):
                self.points=[]
                self.temp=self.img_disp.copy()
                cv2.imshow(self.winname, self.temp)
            elif k==ord('q'):
                self.points=[]
                break
        cv2.destroyWindow(self.winname)
        return [(int(x/self.scale), int(y/self.scale)) for (x,y) in self.points]

def warp_topdown(img, corners_xy, patch_px=220, rows=ROWS, cols=COLS):
    W=cols*patch_px
    H=rows*patch_px
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    src = np.array(corners_xy, np.float32)
    Hm  = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Hm, (W,H), flags=cv2.INTER_CUBIC)
    return warped

def draw_grid_overlay(warped, margin=0.06, rows=ROWS, cols=COLS):
    vis = warped.copy()
    H,W = vis.shape[:2]
    ph=H//rows
    pw=W//cols
    for i in range(rows):
        for j in range(cols):
            y1,y2 = i*ph,(i+1)*ph
            x1,x2 = j*pw,(j+1)*pw
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            dy,dx = int(ph*margin), int(pw*margin)
            cv2.rectangle(vis,(x1+dx,y1+dy),(x2-dx,y2-dy),(255,0,0),2)
    return vis

def compute_patch_means(warped, margin=0.06, rows=ROWS, cols=COLS):
    H,W = warped.shape[:2]
    ph=H//rows
    pw=W//cols
    means=[]
    for i in range(rows):
        for j in range(cols):
            y1,y2 = i*ph,(i+1)*ph
            x1,x2 = j*pw,(j+1)*pw
            dy,dx = int(ph*margin), int(pw*margin)
            roi = warped[y1+dy:y2-dy, x1+dx:x2-dx]
            if roi.ndim==2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            if roi.shape[2]==4:
                roi = roi[:,:,:3]
            b,g,r = [float(v) for v in roi.mean(axis=(0,1))]
            means.append((i,j,r,g,b))
    return means

def main():
    ensure_outdir(OUT_FOLDER)
    csv_path = os.path.join(OUT_FOLDER, CSV_NAME)
    if not os.path.exists(csv_path):
        header=['filename']
        for i in range(ROWS):
            for j in range(COLS):
                header += [f'R_{i}_{j}', f'G_{i}_{j}', f'B_{i}_{j}']
        with open(csv_path,'w',newline='') as f:
            csv.writer(f).writerow(header)
    files = list_images(IN_FOLDER, EXTS)
    if not files:
        print(f"[ERROR] No TIF files found in {IN_FOLDER}")
        return
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"\nProcessing: {base}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("  [WARN] Failed to read, skipped.")
            continue
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2]==4:
            img = img[:,:,:3]
        disp, s = resize_for_display(img, AUTOSCALE_MAX)
        corners = CornerClicker("Pick 4 corners TL→TR→BR→BL", disp, s).run()
        if len(corners)!=4:
            print("  [INFO] Skipped.")
            continue
        warped  = warp_topdown(img, corners, patch_px=PATCH_PX)
        overlay = draw_grid_overlay(warped, margin=MARGIN_FR)
        out_warp = os.path.join(OUT_FOLDER, f"{base}_warped.png")
        out_grid = os.path.join(OUT_FOLDER, f"{base}_grid_overlay.png")
        cv2.imwrite(out_warp,  warped)
        cv2.imwrite(out_grid, overlay)
        means = compute_patch_means(warped, margin=MARGIN_FR)
        row=[base]
        for _,_,r,g,b in means:
            row += [f"{r:.4f}", f"{g:.4f}", f"{b:.4f}"]
        with open(csv_path,'a',newline='') as f:
            csv.writer(f).writerow(row)
        print(f"  ✓ Saved: {out_warp}")
        print(f"  ✓ Saved: {out_grid}")
    print("\nDone.")
    print(f"Output: {OUT_FOLDER}")
    print(f"Data:   {csv_path}")

if __name__=="__main__":
    main()
