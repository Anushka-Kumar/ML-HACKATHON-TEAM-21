"""Out-of-fold (OOF) classifier scores for the 80 TRAINING trips, used to tune the HMM decoder honestly.

Why: if we tuned the HMM on scores from a model that was trained on the same trips, the probabilities would be
over-confident and the tuning would be misleading. With OOF scores every trip is scored by a model that never saw it.

Run (after compare_models.py, and after bayes_opt.py if you ran it):
    python src\\models\\make_oof_scores.py *> results\\oof_run.log
Uses the model/params in results/final_model.json. Output: results/oof_scores_train.parquet
(trip, obs, arc, gps_t, ref_arc, prob) for ALL candidate rows of the training trips.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.dirname(__file__))
from metrics import FC, RESULTS, load_train, load_full  # noqa: E402
from model_zoo import make_model  # noqa: E402

fm = json.load(open(os.path.join(RESULTS, "final_model.json")))
name, params = fm["name"], fm.get("params", {})
print(f"OOF scores with {name} ({fm['variant']}) params={params}")

tr = load_train()                    # training rows (subsampled, filtered)
full = load_full("train")                                                    # ALL rows, used for scoring
trips_all = np.array(sorted(full.trip.unique()))
parts = []
for k, (_, held) in enumerate(GroupKFold(n_splits=5).split(trips_all, groups=trips_all)):
    held_trips = set(trips_all[held])
    m = make_model(name, **params)
    fit_rows = tr[~tr.trip.isin(held_trips)]
    m.fit(fit_rows[FC], fit_rows.y)
    test_rows = full[full.trip.isin(held_trips)]
    p = m.predict_proba(test_rows[FC])[:, 1]
    parts.append(test_rows[["trip", "obs", "arc", "gps_t", "ref_arc"]].assign(prob=p))
    print(f"  fold {k}: fitted on {fit_rows.trip.nunique()} trips, scored {len(held_trips)} held-out trips ({len(test_rows)} rows)", flush=True)

pd.concat(parts).to_parquet(os.path.join(RESULTS, "oof_scores_train.parquet"), index=False)
print("Wrote results/oof_scores_train.parquet")
