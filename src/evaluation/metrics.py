"""Shared evaluation code used by ALL five models (RF, KNN, LR, XGBoost, SVM).

Why one module: every model must be scored with the SAME definitions, on the SAME data split,
otherwise the comparison table is not fair.

What is in here
  1. load_train / load_test : read the shared candidate dataset (one row = GPS observation x candidate arc)
  2. obs_metrics            : task-level metrics (Top-1 accuracy, route continuity)
  3. cls_metrics            : candidate-level metrics (accuracy, precision, recall, macro F1, log loss)
  4. top1_decision_f1       : macro F1 when only the best candidate of each observation is predicted "1"

All five models (and Bayesian optimisation) go through these functions, so the numbers are comparable.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "feature_engineering"))
from data import SEED, load_trip, trips  # noqa: E402  (shared project module)
from phase2_data import feature_cols, load_part  # noqa: E402  (team 8-feature dataset)

FC = feature_cols()                                  # the 8 model input features (team CSV)
SHARED = os.path.join(REPO, "results", "shared")     # where build_shared_dataset.py wrote the parquet files
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")
MODELS_DIR = os.path.join(REPO, "models")
# observations kept per training trip. The default keeps ALL of them (the 5-candidate table is small enough);
# set MAX_OBS_TRAIN=150 to subsample for speed.
MAX_OBS_TRAIN = int(os.environ.get("MAX_OBS_TRAIN", 100000))


# ----------------------------------------------------------------------------- data loading
def load_train(max_obs=None):
    """Training candidates of the 80 train trips (features joined with labels).

    - Keep at most `max_obs` random observations per trip (speed; fixed seed).
    - Label-noise filter (TRAINING ONLY): drop observations whose reference arc is not among
      the 5 candidates (reference routes of a few trips disagree with the GPS points).
    """
    max_obs = MAX_OBS_TRAIN if max_obs is None else max_obs
    tr = load_part("train")
    rng = np.random.default_rng(SEED)
    keep = {t: set(rng.choice(g, min(max_obs, len(g)), replace=False))
            for t, g in tr.groupby("trip")["obs"].unique().items()}
    mask = [o in keep[t] for t, o in zip(tr["trip"].values, tr["obs"].values)]
    tr = tr[mask].reset_index(drop=True)
    has_ref = tr.groupby(["trip", "obs"])["y"].transform("max") == 1
    print(f"[load_train] max_obs/trip={max_obs}: kept {int(has_ref.sum())} of {len(has_ref)} rows after noise filter")
    return tr[has_ref].reset_index(drop=True)


def load_full(part):
    """ALL rows of a part (no subsampling, no filtering), features + labels."""
    return load_part(part)


def load_test():
    """Student-test candidates (10 trips). NEVER filtered: obs without a correct candidate count as wrong."""
    te = load_part("student_test")
    meta = pd.read_csv(os.path.join(SHARED, "per_trip_report_student_test.csv"), dtype={"trip": str})
    arcs = {t: load_trip("student_test", t)[2] for t in trips("student_test")}   # (from_node, to_node) per arc
    return te, meta, arcs


# ----------------------------------------------------------------------------- task-level metrics
def route_continuity(top, arcs_by_trip):
    """Share of consecutive predicted arcs that are the same arc OR connected (to_node of prev == from_node of next)."""
    ok = n = 0
    for t, g in top.groupby("trip"):
        a = g.sort_values("obs").arc.values
        A = arcs_by_trip[t]
        same = a[1:] == a[:-1]
        conn = A[a[:-1], 1] == A[a[1:], 0]
        ok += int((same | conn).sum())
        n += len(a) - 1
    return ok / n


def obs_metrics(df, p, arcs_by_trip=None, n_obs_total=None):
    """Observation-level Top-1 accuracy (+ route continuity) from candidate scores p.

    Top-1 = for each observation take the candidate with the highest score; it is correct if it
    equals the reference arc. Observations with NO correct candidate stay in the denominator
    (n_obs_total) so they count as wrong.
    """
    d = df[["trip", "obs", "arc", "ref_arc"]].copy()
    d["p"] = p
    top = d.loc[d.groupby(["trip", "obs"])["p"].idxmax()]
    denom = n_obs_total if n_obs_total else len(top)
    top1 = (top.arc == top.ref_arc).sum() / denom
    cont = route_continuity(top, arcs_by_trip) if arcs_by_trip is not None else None
    return top1, cont, top


# ----------------------------------------------------------------------------- candidate-level metrics
def cls_metrics(y, p, thr=0.5):
    """Binary metrics on every candidate row at threshold `thr`."""
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat),
                precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                macro_f1=f1_score(y, yhat, average="macro"),
                log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def top1_decision_f1(df, p):
    """Macro F1 and predictions when exactly one candidate per observation (the best) is called '1'."""
    d = df[["trip", "obs", "y"]].reset_index(drop=True)
    d["p"] = p
    idx = d.groupby(["trip", "obs"])["p"].idxmax()
    yhat = np.zeros(len(d), dtype=int)
    yhat[idx.values] = 1
    return f1_score(d.y, yhat, average="macro"), yhat


def confusion(y, yhat):
    return confusion_matrix(y, yhat)
