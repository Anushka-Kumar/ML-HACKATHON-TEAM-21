"""Data loading, projection, candidate generation, feature engineering. Shared by all models."""
import os, numpy as np, pandas as pd
from pyproj import Transformer
from shapely import STRtree, linestrings, points

ROOT = os.environ.get("PATHSNAP_DATA", "data")   # expects data/train and data/student_test
SEED = 42
RADIUS_M = 100.0          # candidate search radius (metres) -- justified by candidate-miss-rate sweep
# Trips come from DIFFERENT cities worldwide (Germany, Ecuador, Netherlands, Poland, USA...), so a single UTM zone
# is wrong. We use a per-trip local Azimuthal-Equidistant projection centred on the road-network centroid:
# distances/bearings are accurate (metres) near the centre, valid for any location on Earth.

def trips(part): return sorted(os.listdir(f"{ROOT}/{part}"))

def load_trip(part, t):
    d = f"{ROOT}/{part}/{t}/{t}"
    track = np.loadtxt(d + ".track")
    nodes = np.loadtxt(d + ".nodes")
    arcs = np.loadtxt(d + ".arcs", dtype=int)
    route = np.loadtxt(d + ".route", dtype=int, ndmin=1) if os.path.exists(d + ".route") else None
    return track, nodes, arcs, route

def make_projector(nodes):
    lon0, lat0 = float(np.mean(nodes[:, 0])), float(np.mean(nodes[:, 1]))
    tf = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)
    def project(lonlat):
        x, y = tf.transform(lonlat[:, 0], lonlat[:, 1]); return np.c_[x, y]
    return project

def bearing(dx, dy): return np.arctan2(dx, dy)  # radians from north

def build_candidates(track, nodes, arcs, radius=RADIUS_M):
    """For each GPS obs return rows (obs_idx, arc_id) of all arcs within `radius` m."""
    project = make_projector(nodes)
    P = project(track[:, :2]); N = project(nodes)
    a = N[arcs[:, 0]]; b = N[arcs[:, 1]]
    geoms = linestrings(np.stack([a, b], axis=1))
    tree = STRtree(geoms)
    q, hit = tree.query(points(P), predicate="dwithin", distance=radius)
    return P, N, a, b, q, hit

def features(track, nodes, arcs, route=None, radius=RADIUS_M):
    """Candidate-level feature table. One row = (GPS observation, candidate arc)."""
    P, N, a, b, q, hit = build_candidates(track, nodes, arcs, radius)
    ab = b[hit] - a[hit]; L = np.hypot(*ab.T) + 1e-9
    ap = P[q] - a[hit]
    t = np.clip((ap * ab).sum(1) / L**2, 0, 1)
    proj = a[hit] + t[:, None] * ab
    dist = np.hypot(*(P[q] - proj).T)
    # signed perpendicular offset (side of road)
    cross = (ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / L
    # vehicle heading from smoothed track (central difference) and speed
    T = track[:, 2]
    dP = np.zeros_like(P); dP[1:-1] = P[2:] - P[:-2]; dP[0] = P[1] - P[0]; dP[-1] = P[-1] - P[-2]
    dT = np.zeros_like(T); dT[1:-1] = T[2:] - T[:-2]; dT[0] = T[1] - T[0]; dT[-1] = T[-1] - T[-2]
    speed = np.hypot(*dP.T) / np.maximum(dT, 1e-3)
    hv = bearing(dP[:, 0], dP[:, 1]); ha = bearing(ab[:, 0], ab[:, 1])
    dh = np.abs(np.angle(np.exp(1j * (hv[q] - ha))))          # 0..pi
    df = pd.DataFrame(dict(obs=q, arc=hit, dist=dist, t_along=t, arc_len=L, side=cross,
                           heading_diff=dh, cos_dh=np.cos(dh), speed=speed[q],
                           dist_to_end=(1 - t) * L, dist_to_start=t * L))
    g = df.groupby("obs")["dist"]
    df["dist_rank"] = g.rank(method="first")
    df["dist_gap_to_best"] = df["dist"] - g.transform("min")
    df["n_cand"] = g.transform("size")
    df["speed_x_cos"] = df["speed"] * df["cos_dh"]
    if route is not None:
        df["y"] = label_candidates(df, P, a, b, route)
    return df, P

def reference_arc_per_obs(P, a, b, route):
    """Reference arc for each GPS obs = nearest arc *within the reference route*, with monotone
    progression along the route (a point can't go back to an earlier route position)."""
    ra = np.unique(route); A = a[ra]; B = b[ra]
    # distance of every obs to every route arc (route is short: 60..2300 arcs)
    ab = B - A; L2 = (ab**2).sum(1) + 1e-9
    D = np.empty((len(P), len(ra)))
    for j in range(len(ra)):
        t = np.clip(((P - A[j]) @ ab[j]) / L2[j], 0, 1)
        D[:, j] = np.hypot(*(P - (A[j] + t[:, None] * ab[j])).T)
    # route position of each unique arc = first index in route sequence
    first = {arc: i for i, arc in enumerate(route)}
    pos = np.array([first[x] for x in ra])
    order = np.argsort(pos); D = D[:, order]; ra = ra[order]
    # monotone DP (Viterbi on route order, forward-only moves) to avoid ambiguity at loops/overlaps
    n, m = D.shape; cost = D.copy(); back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        cm = np.minimum.accumulate(cost[i - 1]); am = np.zeros(m, dtype=np.int32)
        run = 0
        for j in range(m):
            if cost[i - 1, j] <= cm[j] + 1e-12: run = j
            am[j] = run
        cost[i] += cm; back[i] = am
    j = int(np.argmin(cost[-1])); ref = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        ref[i] = ra[j]; j = back[i, j]
    return ref

def label_candidates(df, P, a, b, route):
    ref = reference_arc_per_obs(P, a, b, route)
    df.attrs["ref"] = ref
    return (df["arc"].values == ref[df["obs"].values]).astype(int)

def feature_cols():
    return ["dist", "t_along", "arc_len", "side", "heading_diff", "cos_dh", "speed", "dist_to_end",
            "dist_to_start", "dist_rank", "dist_gap_to_best", "n_cand", "speed_x_cos"]
