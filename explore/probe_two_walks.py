"""Repeatability between two walks of the same space.

Aligns the mid-height wall points of capture B onto capture A with one rigid 2D transform
(coarse rotation search + FFT translation, then trimmed ICP) and measures how far B's walls
sit from A's. Residuals that vary by region after a single rigid fit point to drift.

Usage: python explore/probe_two_walks.py <capture_A> <capture_B>
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

DATA = os.environ.get("COZMO_DATA", "data/captures")
OUT = os.environ.get("COZMO_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
STEP = 10
RES = 0.02


def wall_points(cap):
    cache = os.path.join(OUT, f"{cap}_walls.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xz"], d["fid"]
    o = pd.read_csv(os.path.join(DATA, cap, "odometry.csv"), skipinitialspace=True)
    xs, fs = [], []
    for i in range(0, len(o), STEP):
        r = o.iloc[i]
        f = f"{int(r.frame):06d}.png"
        z = cv2.medianBlur(cv2.imread(os.path.join(DATA, cap, "depth", f), -1).astype(np.float32) / 1000.0, 5)
        c = cv2.imread(os.path.join(DATA, cap, "confidence", f), -1)
        h, w = z.shape
        s = w / 1920.0
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        P = np.stack([(u - r.cx * s) / (r.fx * s) * z, (v - r.cy * s) / (r.fy * s) * z, z], -1)
        dx = P[1:-1, 2:] - P[1:-1, :-2]
        dy = P[2:, 1:-1] - P[:-2, 1:-1]
        n = np.cross(dx, dy)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        zc = z[1:-1, 1:-1]
        ok = (c[1:-1, 1:-1] == 2) & (zc > 0.3) & (zc < 4.0)
        ok &= (np.abs(dx[..., 2]) < 0.04 * zc) & (np.abs(dy[..., 2]) < 0.04 * zc)
        Rw = R.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        pw = P[1:-1, 1:-1][ok] @ Rw.T + np.array([r.x, r.y, r.z])
        nw = n[ok] @ Rw.T
        # vertical surfaces between ~0.9 m below and 0.3 m above the camera
        m = (np.abs(nw[:, 1]) < 0.1) & (pw[:, 1] > r.y - 0.9) & (pw[:, 1] < r.y + 0.3)
        xs.append(pw[m][::3][:, [0, 2]])
        fs.append(np.full(len(pw[m][::3]), i))
    xz, fid = np.concatenate(xs), np.concatenate(fs)
    np.savez(cache, xz=xz, fid=fid)
    return xz, fid


def grid(xz, res=RES):
    cells, counts = np.unique(np.floor(xz / res).astype(np.int64), axis=0, return_counts=True)
    return (cells[counts >= 3] + 0.5) * res


def raster(pts, mn, shape, res=0.05):
    img = np.zeros(shape, np.float32)
    ij = np.floor((pts - mn) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1)
    return np.minimum(img, 5)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def icp(src, dst, Rm, t, iters=40, trim=0.08):
    tree = cKDTree(dst)
    for it in range(iters):
        d, j = tree.query(src @ Rm.T + t)
        m = d < max(trim * (0.9 ** it), 0.03)
        a, b = src[m], dst[j[m]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rn = (U @ Vt).T
        if np.linalg.det(Rn) < 0:
            Vt[-1] *= -1
            Rn = (U @ Vt).T
        Rm, t = Rn, mb - Rn @ ma
    return Rm, t


def main(cap_a, cap_b):
    os.makedirs(OUT, exist_ok=True)
    ga, gb = grid(wall_points(cap_a)[0]), grid(wall_points(cap_b)[0])
    print(f"wall cells: {cap_a} {len(ga)}   {cap_b} {len(gb)}")

    # coarse: rotation every 1 deg, translation by FFT cross-correlation at 5 cm
    mnA = ga.min(0) - 3
    shape = tuple((np.ptp(ga, 0) / 0.05).astype(int) + 121)
    IA = raster(ga, mnA, shape)
    cb = gb.mean(0)
    best = (-1, None, None)
    for deg in np.arange(0, 360, 1.0):
        IB = raster((gb - cb) @ rot(np.radians(deg)).T + ga.mean(0), mnA, shape)
        corr = np.fft.ifft2(np.fft.fft2(IA) * np.conj(np.fft.fft2(IB))).real
        k = np.unravel_index(np.argmax(corr), corr.shape)
        if corr[k] > best[0]:
            sh = np.array([k[0] if k[0] < shape[0] // 2 else k[0] - shape[0],
                           k[1] if k[1] < shape[1] // 2 else k[1] - shape[1]]) * 0.05
            best = (corr[k], deg, sh)
    _, deg, sh = best
    Rm = rot(np.radians(deg))
    t = ga.mean(0) - Rm @ cb + sh
    Rm, t = icp(gb, ga, Rm, t)
    pB = gb @ Rm.T + t
    d, _ = cKDTree(ga).query(pB)
    print(f"rigid fit rotation: {np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])):.2f} deg")
    for thr in (0.02, 0.03, 0.05, 0.10):
        print(f"  {cap_b} wall cells within {int(thr * 100)} cm of {cap_a}: {100 * (d < thr).mean():.1f}%")
    print(f"  median residual of matched walls (<10 cm): {100 * np.median(d[d < 0.10]):.2f} cm")

    tiles = {}
    for p, di in zip(pB, d):
        if di < 0.15:
            tiles.setdefault(tuple(np.floor(p).astype(int)), []).append(di)
    arr = np.array([np.median(v) for v in tiles.values() if len(v) > 30])
    print(f"  per-1m-tile median residual (cm): min {100 * arr.min():.1f}  p50 {100 * np.median(arr):.1f}  "
          f"p90 {100 * np.percentile(arr, 90):.1f}  max {100 * arr.max():.1f}")

    print("  3 m regions re-fitted on their own: region, cells, residual before/after (cm), extra rotation (deg)")
    rows = []
    for key in {tuple(np.floor(p / 3.0).astype(int)) for p in pB}:
        m = np.all(np.floor(pB / 3.0).astype(int) == key, axis=1)
        if m.sum() < 400:
            continue
        R2, t2 = icp(pB[m], ga, np.eye(2), np.zeros(2), iters=25, trim=0.06)
        d2, _ = cKDTree(ga).query(pB[m] @ R2.T + t2)
        d1 = d[m]
        rows.append((key, int(m.sum()), 100 * np.median(d1[d1 < 0.1]), 100 * np.median(d2[d2 < 0.1]),
                     np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))))
    for key, n, before, after, extra in sorted(rows, key=lambda r: -r[1]):
        print(f"    {key} {n:5d}  {before:5.2f} -> {after:5.2f}  {extra:+.2f}")

    # overlay: A blue, B red, overlap black
    mn = np.minimum(ga.min(0), pB.min(0)) - 0.2
    shp = tuple(int(x) for x in ((np.maximum(ga.max(0), pB.max(0)) - mn) / RES).astype(int) + 20)
    img = np.full(shp + (3,), 255, np.uint8)
    ia = np.floor((ga - mn) / RES).astype(int)
    ib = np.floor((pB - mn) / RES).astype(int)
    img[ia[:, 0], ia[:, 1]] = (200, 120, 0)
    img[ib[:, 0], ib[:, 1]] = (0, 0, 220)
    both = np.zeros(shp, bool)
    both[ia[:, 0], ia[:, 1]] = True
    over = np.zeros(shp, bool)
    over[ib[:, 0], ib[:, 1]] = True
    img[both & over] = (0, 0, 0)
    path = os.path.join(OUT, f"overlay_{cap_a}_vs_{cap_b}.png")
    cv2.imwrite(path, img)
    print("wrote", path)


if __name__ == "__main__":
    args = sys.argv[1:] or ["c7d28f72c6", "1a8384c3f6"]
    main(args[0], args[1])
