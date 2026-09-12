"""Normal-aware floor/ceiling heights and wall heading drift for a Stray Scanner capture.

Floor = points facing up well below the camera; ceiling = points facing down above it and
at least 1.9 m over the floor (so furniture tops are never taken as ceiling). Heading drift
is the dominant wall-normal direction (mod 90 deg) per time window.

Usage: python explore/probe_normals.py <capture_id> [<capture_id> ...]
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R

DATA = os.environ.get("COZMO_DATA", "data/captures")
STEP = 10


def frame_points(cap, r):
    f = f"{int(r.frame):06d}.png"
    z = cv2.imread(os.path.join(DATA, cap, "depth", f), -1).astype(np.float32) / 1000.0
    c = cv2.imread(os.path.join(DATA, cap, "confidence", f), -1)
    z = cv2.medianBlur(z, 5)
    h, w = z.shape
    s = w / 1920.0
    fx, fy, cx, cy = r.fx * s, r.fy * s, r.cx * s, r.cy * s
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    Pc = P[1:-1, 1:-1]
    zc = z[1:-1, 1:-1]
    ok = (c[1:-1, 1:-1] == 2) & (zc > 0.3) & (zc < 4.5)
    ok &= (np.abs(dx[..., 2]) < 0.04 * zc) & (np.abs(dy[..., 2]) < 0.04 * zc)
    # orient normals toward the camera
    n[(n * Pc).sum(-1) > 0] *= -1
    Pc, n = Pc[ok][::2], n[ok][::2]
    Rw = R.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
    t = np.array([r.x, r.y, r.z])
    return Pc @ Rw.T + t, n @ Rw.T, Rw @ np.array([0, 0, 1.0])


def peak_refine(y, lo, hi, bw=0.01):
    sel = y[(y > lo) & (y < hi)]
    if len(sel) < 300:
        return None
    hist, e = np.histogram(sel, bins=np.arange(lo, hi + bw, bw))
    c = e[np.argmax(hist)] + bw / 2
    s = sel[np.abs(sel - c) < 0.04]
    for _ in range(3):
        m = np.median(s)
        mad = 1.4826 * np.median(np.abs(s - m)) + 1e-4
        s = s[np.abs(s - m) < 2.5 * mad]
    return float(s.mean()), float(s.std()), len(s)


def dominant_yaw(nrm):
    th = np.degrees(np.arctan2(nrm[:, 2], nrm[:, 0])) % 90
    hist, _ = np.histogram(th, bins=np.arange(0, 90.25, 0.25))
    k = np.argmax(uniform_filter1d(hist.astype(float), 5, mode="wrap"))
    ang = np.radians(th * 4)
    center = np.radians((k * 0.25 + 0.125) * 4)
    d = np.angle(np.exp(1j * (ang - center)))
    m = np.abs(d) < np.radians(12)
    return (np.degrees(center + np.angle(np.exp(1j * d[m]).mean())) / 4) % 90, m.mean()


def main(caps):
    for cap in caps:
        o = pd.read_csv(os.path.join(DATA, cap, "odometry.csv"), skipinitialspace=True)
        idxs = np.arange(0, len(o), STEP)
        Fy, Fid, Cy, Cid, Wn, Wid, up = [], [], [], [], [], [], []
        for i in idxs:
            p, n, fwd = frame_points(cap, o.iloc[i])
            camy = o.iloc[i].y
            up.append(fwd[1])
            fm = (n[:, 1] > 0.97) & (p[:, 1] < camy - 0.7)
            cm = (n[:, 1] < -0.97) & (p[:, 1] > camy + 0.2)
            wm = np.abs(n[:, 1]) < 0.1
            Fy.append(p[fm, 1]); Fid.append(np.full(fm.sum(), i))
            Cy.append(p[cm, 1]); Cid.append(np.full(cm.sum(), i))
            Wn.append(n[wm]); Wid.append(np.full(wm.sum(), i))
        Fy, Fid, Cy, Cid = map(np.concatenate, (Fy, Fid, Cy, Cid))
        Wn, Wid = np.concatenate(Wn), np.concatenate(Wid)
        up = np.array(up)

        print(f"\n===== {cap}")
        seen = np.mean([(Cid == i).sum() > 300 for i in idxs])
        print(f"  frames looking up >20 deg: {100 * (up > 0.34).mean():.1f}%   "
              f"frames with >300 ceiling points: {100 * seen:.1f}%")
        fl = peak_refine(Fy, Fy.min(), Fy.max())
        print(f"  floor  y={fl[0]:.4f}  std={fl[1] * 100:.2f} cm  n={fl[2]}")
        ce = peak_refine(Cy, fl[0] + 1.9, fl[0] + 3.5)
        if ce:
            print(f"  ceil   y={ce[0]:.4f}  std={ce[1] * 100:.2f} cm  n={ce[2]}  "
                  f"-> CEILING HEIGHT {100 * (ce[0] - fl[0]):.2f} cm")
            # several strong peaks usually mean rooms with different ceiling heights
            hist, e = np.histogram(Cy - fl[0], bins=np.arange(1.9, 3.5, 0.02))
            top = [k for k in np.argsort(hist)[::-1][:4] if hist[k] > 0]
            print("  ceiling candidates (height cm: points):",
                  [(round(100 * (e[k] + 0.01), 1), int(hist[k])) for k in top])
        else:
            print("  ceiling not observed")

        half = idxs[len(idxs) // 2]
        for name, a, b in (("1st half", Fid < half, Cid < half), ("2nd half", Fid >= half, Cid >= half)):
            f2 = peak_refine(Fy[a], Fy.min(), Fy.max())
            c2 = peak_refine(Cy[b], fl[0] + 1.9, fl[0] + 3.5)
            if f2 and c2:
                print(f"  {name}: ceiling height {100 * (c2[0] - f2[0]):.2f} cm (floor {100 * (f2[0] - fl[0]):+.2f} cm)")
            else:
                print(f"  {name}: not enough ceiling/floor observations")

        yaws = []
        for wd in np.array_split(idxs, 8):
            m = (Wid >= wd[0]) & (Wid <= wd[-1])
            yaws.append(dominant_yaw(Wn[m]) if m.sum() > 3000 else (np.nan, 0))
        ys = np.array([y for y, _ in yaws])
        print("  dominant wall yaw per window (deg):", np.round(ys, 2))
        print("  heading change vs first window (deg):", np.round((ys - ys[0] + 45) % 90 - 45, 2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["c00a170fe1"])
