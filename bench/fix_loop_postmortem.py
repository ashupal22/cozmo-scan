"""Fix-loop post-mortem (analysis only; docs/fix_loop.md section 6): what drift correction did at tracking breaks on the real walks.
Same fused points, drift correction with the breaks declared (after #2) and without (before), against ARKit:
per segment the true heading error at its first frame (relative to the walk's first frame), the turn applied,
and what is left; over all frames the heading error and the camera position error after one similarity fit."""
import sys, json, numpy as np
ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'bench'))
from video_vs_lidar import upright_video, DATA, DERIVED
from cozmo.ingest.stray import StrayCapture
from cozmo.video.capture import load_video, umeyama
from cozmo.geometry.fusion import fuse
from cozmo.slam.drift import estimate_drift, _segments
import cozmo.video.capture as _vc
_vc.IGNORE_UNSURE_DEPTH = True   # as in the after run (63e77ed), where unsure depth was always ignored
Q = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
wrap = lambda d: (d + 180.0) % 360.0 - 180.0
heading = lambda R: np.degrees(np.arctan2(R[0, 2], R[2, 2]))
for cid in ("c00a170fe1", "1a8384c3f6", "c7d28f72c6"):
    cap = StrayCapture(DATA / cid)
    work = DERIVED / f"{cid}_video_work"
    vcap = load_video(upright_video(cid), work, max_keyframes=None)
    rows = np.load(work / "rows.npy")
    n = len(vcap)
    pts = fuse(vcap)
    h_rec = np.array([heading(vcap.rotation(k)) for k in range(n)])
    h_true = np.array([heading(cap.rotation(rows[k]) @ Q) for k in range(n)])
    err_rec = wrap((h_rec - h_rec[0]) - (h_true - h_true[0]))
    true_pos = cap.positions[rows][:, [0, 2]]
    out = {}
    for name, kw in (("no breaks", dict(jumps=[], relocalized=False)), ("breaks declared", dict(jumps=list(vcap.breaks), relocalized=False))):
        corr, rep = estimate_drift(vcap, pts, **kw)
        theta = np.degrees(corr.theta)
        err = wrap(err_rec + theta - theta[0])
        valid = corr.valid
        P = np.column_stack([corr.positions[:, 0], np.zeros(n), corr.positions[:, 2]])
        T = np.column_stack([true_pos[:, 0], np.zeros(n), true_pos[:, 1]])
        s, R, t = umeyama(P[valid], T[valid])
        perr = np.linalg.norm((s * P[valid] @ R.T + t) - T[valid], axis=1)
        out[name] = (err, rep)
        print(f"{cid} {name:15s}: heading error |median| {np.median(np.abs(err[valid])):5.1f} deg, p90 {np.percentile(np.abs(err[valid]), 90):5.1f}; "
              f"position error median {100*np.median(perr):5.1f} cm (scale {s:.2f}); loops {rep.loops_accepted}/{rep.loops_proposed}, "
              f"rejected by graph {rep.loops_rejected_by_graph}; turns {rep.segment_turns_deg}", flush=True)
    segs = _segments(n, list(vcap.breaks))
    _, rep = out["breaks declared"]
    err_on, _ = out["breaks declared"]
    err_off, _ = out["no breaks"]
    print(f"  {len(segs)} segments; per segment (first frame): recorded error, after (no breaks), after (breaks declared), turn applied")
    for g, (first, last) in enumerate(segs):
        applied = rep.segment_turns_deg.get(g)
        if abs(err_rec[first]) > 20 or applied or abs(err_on[first]) > 20:
            print(f"    segment {g:2d} frames {first:3d}-{last:3d}: recorded {err_rec[first]:+7.1f}, no breaks {err_off[first]:+7.1f}, "
                  f"breaks declared {err_on[first]:+7.1f}, turn {applied}", flush=True)

# Run: COZMO_DATA=/path/to/captures python bench/fix_loop_postmortem.py   (output of the committed run: bench/results/fix_loop/postmortem_drift.log)
