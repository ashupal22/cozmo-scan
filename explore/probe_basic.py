"""Basic look at a Stray Scanner LiDAR capture.

Back-projects confidence-2 depth from every 10th frame into the world frame using the
per-frame intrinsics and ARKit poses, then reports:
  - which camera convention makes the floor a sharp plane
  - floor height and how it drifts across the capture
  - pose jumps (ARKit relocalization)
and writes a top-down map of mid-height points with the walked path.

Usage: python explore/probe_basic.py <capture_id> [<capture_id> ...]
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

DATA = os.environ.get("COZMO_DATA", "data/captures")
OUT = os.environ.get("COZMO_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
STEP = 10


def load(cap):
    return pd.read_csv(os.path.join(DATA, cap, "odometry.csv"), skipinitialspace=True)


def backproject(cap, o, flip, idxs):
    pts, fids = [], []
    for i in idxs:
        r = o.iloc[i]
        f = f"{int(r.frame):06d}.png"
        z = cv2.imread(os.path.join(DATA, cap, "depth", f), -1).astype(np.float32) / 1000.0
        c = cv2.imread(os.path.join(DATA, cap, "confidence", f), -1)
        h, w = z.shape
        s = w / 1920.0
        fx, fy, cx, cy = r.fx * s, r.fy * s, r.cx * s, r.cy * s
        v, u = np.nonzero((c == 2) & (z > 0.2) & (z < 5.0))
        zz = z[v, u]
        p = np.stack([(u + 0.5 - cx) / fx * zz, (v + 0.5 - cy) / fy * zz, zz], 1)
        if flip:  # OpenCV (x right, y down, z forward) -> ARKit (x right, y up, z back)
            p = p * np.array([1, -1, -1])
        Rw = R.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        pw = p @ Rw.T + np.array([r.x, r.y, r.z])
        pts.append(pw[::4])
        fids.append(np.full(len(pw[::4]), i))
    return np.concatenate(pts), np.concatenate(fids)


def height_hist(y, bw=0.01):
    lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
    return np.histogram(y, bins=np.arange(lo, hi, bw))


def refine(y, center, win=0.05):
    sel = y[np.abs(y - center) < win]
    for _ in range(3):
        m = np.median(sel)
        mad = np.median(np.abs(sel - m)) + 1e-6
        sel = sel[np.abs(sel - m) < 3 * 1.4826 * mad]
    return float(np.mean(sel)), float(np.std(sel)), len(sel)


def floor_height(y):
    hist, e = height_hist(y)
    c = (e[:-1] + e[1:]) / 2
    return refine(y, c[np.argmax(hist[: len(c) // 2])])


def main(caps):
    os.makedirs(OUT, exist_ok=True)
    for cap in caps:
        o = load(cap)
        idxs = np.arange(0, len(o), STEP)
        print(f"\n===== {cap}  ({len(idxs)} keyframes)")

        best = None
        for flip in (False, True):
            p, _ = backproject(cap, o, flip, idxs[::5])
            hist, _ = height_hist(p[:, 1])
            sharp = np.sort(hist)[-2:].sum() / hist.sum()
            print(f"  convention flip={flip}: top-2 height-bin mass = {sharp:.3f}")
            if best is None or sharp > best[0]:
                best = (sharp, flip)

        P, F = backproject(cap, o, best[1], idxs)
        fm, fs, fn = floor_height(P[:, 1])
        print(f"  floor y={fm:.4f}  per-point std {fs * 100:.1f} cm  n={fn}")

        fh = []
        for wd in np.array_split(idxs, 6):
            y = P[(F >= wd[0]) & (F <= wd[-1]), 1]
            fh.append(refine(y, fm, 0.08)[0] if (np.abs(y - fm) < 0.08).sum() > 500 else np.nan)
        print("  floor height per time window (cm, vs overall):", np.round((np.array(fh) - fm) * 100, 1))

        pos = o[["x", "y", "z"]].values
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        jumps = np.nonzero(step > 0.05)[0]
        print("  pose jumps > 5 cm (frame, m):", [(int(j), round(float(step[j]), 3)) for j in jumps])

        # top-down map of mid-height points, walked path in red
        W = (P[:, 1] > fm + 0.3) & (P[:, 1] < fm + 2.0)
        xz = P[W][:, [0, 2]]
        res = 0.02
        mn = xz.min(0)
        ij = ((xz - mn) / res).astype(int)
        img = np.zeros(ij.max(0) + 1, np.float32)
        np.add.at(img, (ij[:, 0], ij[:, 1]), 1)
        im = np.log1p(img)
        im = (255 * im / im.max()).astype(np.uint8)
        col = cv2.cvtColor(255 - im, cv2.COLOR_GRAY2BGR)
        tr = ((pos[:, [0, 2]] - mn) / res).astype(int)
        for a, b in zip(tr[:-1:5], tr[5::5]):
            cv2.line(col, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), (0, 0, 255), 1)
        path = os.path.join(OUT, f"{cap}_topdown.png")
        cv2.imwrite(path, col)
        print("  wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["c00a170fe1"])
