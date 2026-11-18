"""Check that phase2_data.compute_features (used by the API) reproduces the team CSV exactly.

Run:  python src/feature_engineering/check_feature_port.py [part] [trip ...]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import CSV_DIR, FEATURES, compute_features  # noqa: E402
from data import load_trip  # noqa: E402

part = sys.argv[1] if len(sys.argv) > 1 else "student_test"
trip_ids = sys.argv[2:] or ["00000052", "00000009", "00000056"]
for t in trip_ids:
    track, nodes, arcs, _ = load_trip(part, t)
    mine, _ = compute_features(track, nodes, arcs)
    theirs = pd.read_csv(os.path.join(CSV_DIR[part], f"{t}_expanded_candidate_features.csv"),
                         dtype={"trip_id": str})
    same_rows = (len(mine) == len(theirs) and (mine.obs.values == theirs.observation_id.values).all()
                 and (mine.arc.values == theirs.candidate_arc_id.values).all())
    worst = {}
    if same_rows:
        for c in FEATURES:
            a, b = mine[c].values.astype(float), theirs[c].values.astype(float)
            nan_ok = (np.isnan(a) == np.isnan(b)).all()
            diff = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
            worst[c] = (nan_ok, float(diff))
    ok = same_rows and all(n and d < 1e-6 for n, d in worst.values())
    print(f"trip {t}: rows {len(mine)} vs {len(theirs)} | same obs+arc order: {same_rows} | {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("   per-feature (nan pattern equal, max abs diff):", worst)
