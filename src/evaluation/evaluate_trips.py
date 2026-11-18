"""Run the FINAL pipeline (saved model + saved HMM parameters) on a folder of trips and report accuracy.

Inference only: nothing is trained or tuned here. Use it once on the evaluator trips and do NOT change the model,
features or decoder parameters afterwards based on these numbers.

Run from the repo root:
    python src/evaluation/evaluate_trips.py evaluator_test
Output: results/<folder>_results.csv and a printed table.
"""
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
folder = sys.argv[1] if len(sys.argv) > 1 else "evaluator_test"
# the API pipeline reads <PATHSNAP_DATA>/<EVAL_PART>/<trip>/<trip>.*  ->  repo root + folder name
os.environ["PATHSNAP_DATA"] = REPO
os.environ["EVAL_PART"] = folder
sys.path.insert(0, os.path.join(REPO, "backend"))
from pipeline import Pipeline  # noqa: E402

pipe = Pipeline()
rows = []
for t in pipe.trips():
    r = pipe.run(t)
    e = r["evaluation"]
    rows.append(dict(trip=t, n_obs=r["n_gps_points"], candidate_miss_rate=e["candidate_miss_rate"],
                     clf_top1=e["clf"]["top1"], hmm_top1=e["hmm"]["top1"],
                     clf_route_f1=e["clf"]["route_f1"], hmm_route_f1=e["hmm"]["route_f1"],
                     hmm_route_precision=e["hmm"]["route_precision"], hmm_route_recall=e["hmm"]["route_recall"],
                     clf_continuity=e["clf"]["route_continuity"], hmm_continuity=e["hmm"]["route_continuity"],
                     seconds=r["timing"]["total_s"]))
    print(f"trip {t}: top1 {rows[-1]['clf_top1']:.4f} -> {rows[-1]['hmm_top1']:.4f} | route F1 {rows[-1]['hmm_route_f1']:.4f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(REPO, "results", f"{folder}_results.csv"), index=False)
w = df.n_obs / df.n_obs.sum()
print(f"\n{folder}: {len(df)} trips, {df.n_obs.sum()} observations (model: {pipe.model_name})")
print("observation-weighted: candidate miss %.4f | Top-1 classifier %.4f -> HMM %.4f" % (
    (w * df.candidate_miss_rate).sum(), (w * df.clf_top1).sum(), (w * df.hmm_top1).sum()))
print("mean over trips: route F1 classifier %.4f -> HMM %.4f | HMM precision %.4f recall %.4f | continuity %.4f" % (
    df.clf_route_f1.mean(), df.hmm_route_f1.mean(), df.hmm_route_precision.mean(), df.hmm_route_recall.mean(), df.hmm_continuity.mean()))
