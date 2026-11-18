"""Tune the HMM decoder, then compare classifier-only vs classifier + HMM on the student-test trips.

Run:  python src\\hmm\\run_hmm.py *> results\\hmm_run.log
Stages (env HMM_STAGE = tune | eval | all, default all):
  tune : grid search of decoder parameters on OUT-OF-FOLD scores of a sample of TRAINING trips
         (needs results/oof_scores_train.parquet from make_oof_scores.py). Student-test is not used here.
  eval : decode the 10 student-test trips with the chosen parameters and compare with the classifier alone.
Env: TUNE_TRIPS (default 12) trips sampled, TUNE_MAX_OBS (default 1500) first observations per trip.

Everything printed/saved comes from running this code; nothing is typed in.
"""
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
from metrics import RESULTS, FIGURES, SHARED, SEED, load_trip, load_full  # noqa: E402
from viterbi_hmm import DEFAULT_PARAMS, TripHMM, classifier_only, score_prediction  # noqa: E402

STAGE = os.environ.get("HMM_STAGE", "all")
PARAMS_JSON = os.path.join(RESULTS, "hmm_best_params.json")
GRID = dict(lam=[0.5, 1.0, 2.0], disc_penalty=[4.0, 10.0], max_hops=[4], gap_s=[30.0], emission_power=[0.5, 1.0, 2.0])


def ref_array(g, n_obs):
    """Reference arc per observation (-1 where the observation has no candidate rows)."""
    ref = np.full(n_obs, -1, dtype=np.int64)
    r = g.drop_duplicates("obs")
    ref[r.obs.values] = r.ref_arc.values
    return ref


def aggregate(rows):
    d = pd.DataFrame(rows)
    out = dict(top1=d.correct.sum() / d.n_obs.sum(), route_continuity=d.cont_ok.sum() / max(d.cont_n.sum(), 1))
    for c in ("route_precision", "route_recall", "route_f1"):
        if c in d:
            out[c] = d[c].mean()            # mean over trips (each trip has one route)
    return out


# ----------------------------------------------------------------------------- stage 1: tune on training OOF scores
def tune():
    path = os.path.join(RESULTS, "oof_scores_train.parquet")
    if not os.path.exists(path):
        sys.exit("Missing results/oof_scores_train.parquet. Run src\\models\\make_oof_scores.py first.")
    oof = pd.read_parquet(path)
    meta = pd.read_csv(os.path.join(SHARED, "per_trip_report_train.csv"), dtype={"trip": str}).set_index("trip")
    rng = np.random.default_rng(SEED)
    trips = sorted(rng.choice(sorted(oof.trip.unique()), int(os.environ.get("TUNE_TRIPS", 12)), replace=False))
    max_obs = int(os.environ.get("TUNE_MAX_OBS", 1500))
    prepared = []
    for t in trips:
        g = oof[(oof.trip == t) & (oof.obs < max_obs)]               # first max_obs observations (contiguous in time)
        n = min(max_obs, int(meta.loc[t, "n_obs"]))
        arcs = load_trip("train", t)[2]
        prepared.append((TripHMM(arcs, g), ref_array(g, n), n, arcs))
    print(f"Tuning on {len(trips)} training trips, <= {max_obs} obs each (out-of-fold scores)")

    rows = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        res = [score_prediction(h.decode(p), ref, n, arcs) for h, ref, n, arcs in prepared]
        r = dict(**p, **aggregate(res))
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).sort_values(["top1", "route_continuity"], ascending=False)
    df.to_csv(os.path.join(RESULTS, "hmm_tuning.csv"), index=False)
    best = {k: (float(df.iloc[0][k]) if k != "max_hops" else int(df.iloc[0][k])) for k in keys}
    json.dump(best, open(PARAMS_JSON, "w"), indent=2)

    base = []                                   # classifier-only on the same sample, for reference
    for t, (_, ref, n, arcs) in zip(trips, prepared):
        g = oof[(oof.trip == t) & (oof.obs < max_obs)]
        base.append(score_prediction(classifier_only(g), ref, n, arcs))
    print("classifier-only on the same training sample:", {k: round(v, 4) for k, v in aggregate(base).items()})
    print("best decoder params:", best)


# ----------------------------------------------------------------------------- stage 2: evaluate on student-test
def evaluate():
    fm = json.load(open(os.path.join(RESULTS, "final_model.json")))
    params = json.load(open(PARAMS_JSON)) if os.path.exists(PARAMS_JSON) else DEFAULT_PARAMS
    print(f"Evaluating with model {fm['name']} ({fm['variant']}), decoder params {params}")
    te = load_full("student_test")[["trip", "obs", "arc", "gps_t", "ref_arc"]]
    sc = pd.read_csv(os.path.join(REPO_ROOT, fm["scores_path"]), dtype={"trip": str})
    assert (sc.arc.values == te.arc.values).all() and (sc.obs.values == te.obs.values).all(), "scores/rows misaligned"
    te["prob"] = sc.prob.values
    meta = pd.read_csv(os.path.join(SHARED, "per_trip_report_student_test.csv"), dtype={"trip": str}).set_index("trip")

    clf_rows, hmm_rows, pred_rows, per_trip = [], [], [], []
    for t, g in te.groupby("trip"):
        _, _, arcs, route = load_trip("student_test", t)
        n = int(meta.loc[t, "n_obs"])
        ref = ref_array(g, n)
        clf = classifier_only(g)
        hmm = TripHMM(arcs, g).decode(params)
        c, h = score_prediction(clf, ref, n, arcs, route), score_prediction(hmm, ref, n, arcs, route)
        clf_rows.append(c)
        hmm_rows.append(h)
        per_trip.append(dict(trip=t, n_obs=n, clf_top1=c["top1"], hmm_top1=h["top1"],
                             clf_route_f1=c["route_f1"], hmm_route_f1=h["route_f1"]))
        pred_rows.append(clf.rename(columns={"arc": "arc_clf"}).merge(hmm.rename(columns={"arc": "arc_hmm"}), on="obs", how="outer").assign(trip=t))
        print(f"  trip {t}: top1 classifier {c['top1']:.4f} -> HMM {h['top1']:.4f}", flush=True)

    summary = pd.DataFrame([dict(method="classifier_only", **aggregate(clf_rows)),
                            dict(method="classifier+HMM", **aggregate(hmm_rows))])
    summary.to_csv(os.path.join(RESULTS, "hmm_comparison.csv"), index=False)
    pd.DataFrame(per_trip).to_csv(os.path.join(RESULTS, "hmm_per_trip.csv"), index=False)
    pd.concat(pred_rows)[["trip", "obs", "arc_clf", "arc_hmm"]].to_csv(
        os.path.join(RESULTS, "hmm_predictions_student_test.csv"), index=False)
    print(summary.round(4).to_string(index=False))

    os.makedirs(FIGURES, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a_, c, ttl in zip(ax, ["top1", "route_continuity", "route_f1"],
                          ["Pointwise Top-1", "Route continuity", "Route F1 (arc set vs reference route)"]):
        a_.bar(summary.method, summary[c], color=["#e9c46a", "#2a6f97"])
        a_.set_title(ttl)
        a_.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "hmm_vs_classifier.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    if STAGE in ("tune", "all"):
        tune()
    if STAGE in ("eval", "all"):
        evaluate()
