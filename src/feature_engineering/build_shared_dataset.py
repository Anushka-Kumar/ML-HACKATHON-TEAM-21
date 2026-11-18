"""Shared candidate-level dataset for the whole team (ALL observations, no subsampling).
Outputs -> results/shared/ : candidates_<part>.parquet, per_trip_report_<part>.csv, feature_schema.csv, execution_log.txt
Run: python src/build_shared_dataset.py"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from scipy.spatial import cKDTree
from data import *
OUT = "results/shared"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/execution_log.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

def midpoint_baseline_miss(P, a, b, ref, k_mid=30, k_keep=5):
    """Re-implements Person 1's method: 30 nearest segment MIDPOINTS -> keep 5 nearest by point-to-segment distance."""
    mids = (a + b) / 2; kk = min(k_mid, len(mids)); _, idx = cKDTree(mids).query(P, k=kk)
    idx = np.atleast_2d(idx.T).T if kk == 1 else idx
    A = a[idx]; B = b[idx]; ab = B - A; L2 = (ab**2).sum(-1) + 1e-9
    t = np.clip(((P[:, None, :] - A) * ab).sum(-1) / L2, 0, 1)
    d = np.hypot(*np.moveaxis(P[:, None, :] - (A + t[..., None] * ab), -1, 0))
    keep = np.take_along_axis(idx, np.argsort(d, axis=1)[:, :k_keep], axis=1)
    return 1 - (keep == ref[:, None]).any(1).mean()

SCHEMA = [
 ("trip", "id", "Trip folder id"), ("obs", "id", "0-based GPS observation index within the trip (row of .track)"),
 ("gps_t", "s", "GPS timestamp (seconds) from .track"), ("arc", "id", "Candidate road segment id = 0-based row of .arcs (preserved for connectivity checks)"),
 ("ref_arc", "id", "Reference arc for this observation (monotone alignment to .route) - ground-truth proxy, NOT a model input"),
 ("y", "label", "1 if candidate arc == ref_arc else 0"),
 ("dist", "m", "Point-to-segment distance GPS point -> candidate arc (per-trip local metric projection)"),
 ("t_along", "0-1", "Fractional position of the projected point along the arc"),
 ("arc_len", "m", "Candidate arc length"), ("side", "m", "Signed perpendicular offset (left/right of arc direction)"),
 ("heading_diff", "rad", "|vehicle heading - arc bearing|, wrapped to 0..pi"), ("cos_dh", "-1..1", "cos(heading_diff)"),
 ("speed", "m/s", "Speed from central difference of the track"), ("dist_to_end", "m", "Distance along arc from projected point to arc end"),
 ("dist_to_start", "m", "Distance along arc from arc start to projected point"),
 ("dist_rank", "rank", "Rank of candidate by dist within its observation (1 = nearest)"),
 ("dist_gap_to_best", "m", "dist minus the smallest dist among this observation's candidates"),
 ("n_cand", "count", "Number of candidates for this observation"), ("speed_x_cos", "m/s", "speed * cos_dh (moving along the arc)")]
pd.DataFrame(SCHEMA, columns=["column", "unit", "description"]).to_csv(f"{OUT}/feature_schema.csv", index=False)
say(f"candidate rule: all arcs within {RADIUS_M} m of the GPS point (point-to-segment distance); projection: per-trip local AEQD")
for part in ["train", "student_test"]:
    frames, rep = [], []; T0 = time.time()
    for t in trips(part):
        t0 = time.time(); tr, nd, ar, rt = load_trip(part, t)
        df, P = features(tr, nd, ar, rt); ref = df.attrs["ref"]; n = len(tr)
        pr = make_projector(nd); N = pr(nd); a = N[ar[:, 0]]; b = N[ar[:, 1]]
        pos = df[df.y == 1].drop_duplicates("obs").set_index("obs").dist_rank.reindex(range(n))
        rep.append(dict(trip=t, n_obs=n, n_rows=len(df), mean_cands_per_obs=len(df) / n,
            miss_radius=float(pos.isna().mean()), miss_nearest5=float((~(pos <= 5)).mean()), miss_nearest10=float((~(pos <= 10)).mean()),
            miss_nearest20=float((~(pos <= 20)).mean()), miss_midpoint30to5_P1method=float(midpoint_baseline_miss(P, a, b, ref)),
            pos_rate=float(df.y.mean()), n_nodes=len(nd), n_arcs=len(ar), n_route_arcs=len(np.unique(rt)),
            gps_dt_max_s=float(np.diff(tr[:, 2]).max()), seconds=round(time.time() - t0, 1)))
        df["trip"] = t; df["gps_t"] = tr[df["obs"].values, 2]; df["ref_arc"] = ref[df["obs"].values]; df.attrs = {}
        frames.append(df); say(part, t, "obs", n, "rows", len(df), "miss@radius %.4f  nearest5 %.4f  P1-midpoint %.4f" % (rep[-1]["miss_radius"], rep[-1]["miss_nearest5"], rep[-1]["miss_midpoint30to5_P1method"]))
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if out[c].dtype == "float64" and c not in ("gps_t",): out[c] = out[c].astype("float32")
    out = out[["trip", "obs", "gps_t", "arc", "ref_arc", "y"] + [c for c in out.columns if c not in ("trip", "obs", "gps_t", "arc", "ref_arc", "y")]]
    out.to_parquet(f"{OUT}/candidates_{part}.parquet"); r = pd.DataFrame(rep); r.to_csv(f"{OUT}/per_trip_report_{part}.csv", index=False)
    w = r.n_obs / r.n_obs.sum()
    say(f"== {part}: trips {len(r)} obs {r.n_obs.sum()} rows {len(out)} | obs-weighted miss: radius {np.dot(w, r.miss_radius):.4f}, nearest5 {np.dot(w, r.miss_nearest5):.4f}, nearest10 {np.dot(w, r.miss_nearest10):.4f}, nearest20 {np.dot(w, r.miss_nearest20):.4f}, P1 midpoint30->5 {np.dot(w, r.miss_midpoint30to5_P1method):.4f} | {time.time() - T0:.0f}s")
