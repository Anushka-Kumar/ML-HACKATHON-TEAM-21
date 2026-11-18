"""Phase-2 candidate features (the team's 8-feature dataset) for the models, HMM and API.

Two jobs
  1. build_part(part): turn Person 1's per-trip CSVs (src/data_processing/improved_noise_features.py output)
     into parquet files for the models.  The feature files contain NO labels. Labels are made separately
     (labels_<part>.parquet) by aligning the GPS track to the reference route, and are joined only inside the
     training / evaluation code.
  2. compute_features(track, nodes, arcs): the same feature computation for ONE trip straight from the raw files,
     used by the API for trips that have no CSV (e.g. the hidden evaluation trips). It re-implements the logic of
     improved_noise_features.py (it imports its noise screening and helper functions) and is checked against the
     CSVs by src/feature_engineering/check_feature_port.py.

The 8 features (names as in the team CSV)
  distance_to_arc_m       point-to-segment distance GPS point -> candidate arc
  heading_difference_deg  |GPS heading - road heading| (NaN when the vehicle is (nearly) stationary)
  candidate_rank          1..5, rank by distance
  gps_speed_mps           speed from the cleaned track (backward difference)
  gps_acceleration_mps2   change of speed per second
  gps_turn_angle_deg      change of heading between consecutive steps
  road_segment_length_m   length of the candidate arc
  network_connectivity    1 if the candidate arc starts where one of the previous observation's candidates ends
Candidates per observation: the 5 nearest arcs (found among the 100 nearest arc midpoints).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src", "data_processing"))
from data import load_trip, trips, reference_arc_per_obs  # noqa: E402
from improved_noise_features import (screen_and_smooth, bearing, angle_diff,  # noqa: E402
                                     TOP_CANDIDATES, MIDPOINT_SEARCH_K)
from scipy.spatial import cKDTree  # noqa: E402

FEATURES = ["distance_to_arc_m", "heading_difference_deg", "candidate_rank", "gps_speed_mps",
            "gps_acceleration_mps2", "gps_turn_angle_deg", "road_segment_length_m", "network_connectivity"]
SHARED = os.path.join(REPO, "results", "shared")
CSV_DIR = {"train": os.path.join(REPO, "results", "phase2"),
           "student_test": os.path.join(REPO, "results", "phase2_student_test")}


def feature_cols():
    return list(FEATURES)


def _utm(lon, lat):
    """Same projection choice as improved_noise_features.py: UTM zone of the trip's mean position."""
    zone = int((np.mean(lon) + 180) // 6) + 1
    epsg = (32600 if np.mean(lat) >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)


# ----------------------------------------------------------------------------- features from raw files
def compute_features(track, nodes, arcs):
    """Candidate-feature table for one trip from .track/.nodes/.arcs only (no route, no labels).

    Returns (df, aux). df columns: obs, arc, gps_t + FEATURES. aux = dict(P, a, b): cleaned GPS points and arc
    end points in the trip's UTM projection (used by the evaluation step to align to the reference route).
    """
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    tf = _utm(lon, lat)
    gx, gy = tf.transform(lon, lat)
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    gx_c, gy_c, _noise, _dev = screen_and_smooth(gx, gy, ts)      # flag + interpolate isolated spikes
    n = len(track)

    # speed, heading, acceleration, turn angle (all from the cleaned track, backward differences)
    speed = np.full(n, np.nan)
    dt = np.diff(ts)
    step = np.hypot(np.diff(gx_c), np.diff(gy_c))
    valid_dt = dt > 0
    speed[1:][valid_dt] = step[valid_dt] / dt[valid_dt]
    heading = np.full(n, np.nan)
    ok = valid_dt & (dt <= 5) & (step > 0.5)
    heading[1:][ok] = bearing(gx_c[:-1][ok], gy_c[:-1][ok], gx_c[1:][ok], gy_c[1:][ok])
    accel = np.full(n, np.nan)
    el = np.diff(ts)
    ok = (el > 0) & (el <= 5) & np.isfinite(speed[1:]) & np.isfinite(speed[:-1])
    ok[0] = False                                                  # acceleration starts at observation 2
    accel[1:][ok] = (speed[1:][ok] - speed[:-1][ok]) / el[ok]
    turn = np.full(n, np.nan)
    ok = np.isfinite(heading[1:]) & np.isfinite(heading[:-1])
    ok[0] = False
    turn[1:][ok] = angle_diff(heading[1:][ok], heading[:-1][ok])

    # candidate arcs: 5 nearest by point-to-segment distance among the 100 nearest arc midpoints
    valid = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    arc_ids = np.flatnonzero(valid)
    va = arcs[valid]
    sx, sy, ex, ey = nx[va[:, 0]], ny[va[:, 0]], nx[va[:, 1]], ny[va[:, 1]]
    tree = cKDTree(np.column_stack(((sx + ex) / 2, (sy + ey) / 2)))
    seg_len = np.hypot(ex - sx, ey - sy)
    road_head = bearing(sx, sy, ex, ey)
    k = min(MIDPOINT_SEARCH_K, len(va))
    _, IX = tree.query(np.column_stack((gx_c, gy_c)), k=k)
    IX = np.sort(IX.reshape(n, k), axis=1)                        # ascending arc index so ties break by index
    dx, dy = ex[IX] - sx[IX], ey[IX] - sy[IX]
    den = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(den == 0, 0.0, np.clip(((gx_c[:, None] - sx[IX]) * dx + (gy_c[:, None] - sy[IX]) * dy) / den, 0, 1))
    dist = np.hypot(gx_c[:, None] - (sx[IX] + f * dx), gy_c[:, None] - (sy[IX] + f * dy))
    order = np.argsort(dist, axis=1, kind="stable")[:, :TOP_CANDIDATES]
    J = np.take_along_axis(IX, order, axis=1)                    # (n, <=5) indices into `va`
    D = np.take_along_axis(dist, order, axis=1)

    rows_obs, rows = [], []
    prev_to = set()
    for obs in range(n):
        cur_to = set()
        for rank in range(J.shape[1]):
            j = int(J[obs, rank])
            from_n, to_n = int(va[j, 0]), int(va[j, 1])
            conn = int(from_n in prev_to) if obs > 0 else 0
            hd = float(angle_diff(heading[obs], road_head[j])) if np.isfinite(heading[obs]) else np.nan
            rows.append((obs, int(arc_ids[j]), ts[obs], float(D[obs, rank]), hd, rank + 1, speed[obs], accel[obs],
                         turn[obs], float(seg_len[j]), conn))
            cur_to.add(to_n)
        prev_to = cur_to
    df = pd.DataFrame(rows, columns=["obs", "arc", "gps_t"] + FEATURES)
    return df, dict(P=np.column_stack((gx_c, gy_c)), a=np.column_stack((nx[arcs[:, 0]], ny[arcs[:, 0]])),
                    b=np.column_stack((nx[arcs[:, 1]], ny[arcs[:, 1]])))


# ----------------------------------------------------------------------------- CSV -> parquet (+ labels)
def _read_trip_csv(path):
    cols = ["trip_id", "observation_id", "candidate_arc_id", "timestamp",
            "gps_longitude_processed", "gps_latitude_processed"] + FEATURES
    d = pd.read_csv(path, usecols=cols, dtype={"trip_id": str})
    return d


def build_part(part):
    """results/phase2*/<trip>_expanded_candidate_features.csv -> features/labels parquet + per-trip report."""
    files = sorted(f for f in glob.glob(os.path.join(CSV_DIR[part], "*_expanded_candidate_features.csv"))
                   if not os.path.basename(f).startswith("all_trips"))
    if not files:
        sys.exit(f"No per-trip CSVs in {CSV_DIR[part]}. Run src/data_processing/improved_noise_features.py first.")
    feats, labs, rep = [], [], []
    for f in files:
        d = _read_trip_csv(f)
        t = d.trip_id.iloc[0]
        track, nodes, arcs, route = load_trip(part, t)
        # cleaned GPS point of every observation (identical on all 5 candidate rows) -> same UTM projection
        o = d.drop_duplicates("observation_id").sort_values("observation_id")
        n = len(track)
        assert len(o) == n and (o.observation_id.values == np.arange(n)).all(), f"trip {t}: observations missing"
        tf = _utm(track[:, 0], track[:, 1])
        P = np.column_stack(tf.transform(o.gps_longitude_processed.values, o.gps_latitude_processed.values))
        nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
        N = np.column_stack((nx, ny))
        ref = reference_arc_per_obs(P, N[arcs[:, 0]], N[arcs[:, 1]], route)   # reference arc per observation
        x = d.rename(columns={"trip_id": "trip", "observation_id": "obs", "candidate_arc_id": "arc",
                              "timestamp": "gps_t"})[["trip", "obs", "arc", "gps_t"] + FEATURES]
        feats.append(x)
        labs.append(pd.DataFrame(dict(trip=t, obs=np.arange(n), ref_arc=ref)))
        hit = (x.arc.values == ref[x.obs.values])
        found = np.zeros(n, dtype=bool)
        found[x.obs.values[hit]] = True
        rep.append(dict(trip=t, n_obs=n, n_rows=len(x), miss_rate=float(1 - found.mean()), pos_rate=float(hit.mean()),
                        n_nodes=len(nodes), n_arcs=len(arcs), n_route_arcs=len(np.unique(route)),
                        gps_dt_max_s=float(np.diff(track[:, 2]).max())))
        print(f"{part} {t}: obs {n} rows {len(x)} candidate miss rate {rep[-1]['miss_rate']:.4f}", flush=True)
    os.makedirs(SHARED, exist_ok=True)
    pd.concat(feats, ignore_index=True).to_parquet(os.path.join(SHARED, f"features_{part}.parquet"), index=False)
    pd.concat(labs, ignore_index=True).to_parquet(os.path.join(SHARED, f"labels_{part}.parquet"), index=False)
    r = pd.DataFrame(rep)
    r.to_csv(os.path.join(SHARED, f"per_trip_report_{part}.csv"), index=False)
    w = r.n_obs / r.n_obs.sum()
    print(f"== {part}: trips {len(r)} obs {r.n_obs.sum()} rows {r.n_rows.sum()} | obs-weighted candidate miss rate {np.dot(w, r.miss_rate):.4f}")


def load_part(part):
    """Features + labels joined (the only place where labels meet features). y = 1 if candidate == reference arc."""
    f = pd.read_parquet(os.path.join(SHARED, f"features_{part}.parquet"))
    l = pd.read_parquet(os.path.join(SHARED, f"labels_{part}.parquet"))
    df = f.merge(l, on=["trip", "obs"], how="left")
    df["y"] = (df["arc"] == df["ref_arc"]).astype(int)
    return df


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["train", "student_test"]):
        build_part(p)
