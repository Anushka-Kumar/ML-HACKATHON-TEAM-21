"""Train/evaluate loop shared by models_p3.py and bayes_opt.py (same protocol as models_p2.py).

Protocol (identical for every model):
  1. GroupKFold(5) on the 80 training trips: a trip is never split between train and validation folds,
     so the validation score measures generalisation to UNSEEN trips.
  2. Final fit on all 80 training trips -> evaluate on the 10 student-test trips (never used to fit anything).
  3. Save: result CSVs, per-trip Top-1, joblib model, scores file (trip, obs, arc, prob) for the HMM, figures.
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from metrics import (FC, SEED, RESULTS, FIGURES, MODELS_DIR, MAX_OBS_TRAIN, load_train, load_test,
                     obs_metrics, cls_metrics, top1_decision_f1, confusion)


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


TRAIN_SCORE_ROWS = 100_000   # training-side scores (over/underfitting check) use at most this many random rows


def cv_eval(make, tr, n_splits=5, verbose=True):
    """Group-aware cross-validation. Returns (summary dict, list of per-fold dicts).

    Validation scores use ALL held-out rows. The training-side scores (only for the over/underfitting check) use a
    fixed random sample of up to TRAIN_SCORE_ROWS training rows: scoring every training row is very slow for KNN
    (a neighbour search per row) and a sample estimates the same train score.
    """
    folds = []
    rng = np.random.default_rng(SEED)
    for k, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(tr, groups=tr.trip)):
        m = make()
        m.fit(tr.iloc[a][FC], tr.iloc[a].y)
        pv = m.predict_proba(tr.iloc[b][FC])[:, 1]      # held-out trips
        a = np.sort(rng.choice(a, min(len(a), TRAIN_SCORE_ROWS), replace=False))   # sample of training rows
        pt = m.predict_proba(tr.iloc[a][FC])[:, 1]      # training trips (for the over/underfitting check)
        top1, _, _ = obs_metrics(tr.iloc[b], pv)
        f = dict(fold=k,
                 val_macro_f1=f1_score(tr.iloc[b].y, (pv >= .5).astype(int), average="macro"),
                 train_macro_f1=f1_score(tr.iloc[a].y, (pt >= .5).astype(int), average="macro"),
                 val_top1=top1,
                 val_logloss=log_loss(tr.iloc[b].y, _clip(pv)),
                 train_logloss=log_loss(tr.iloc[a].y, _clip(pt)))
        folds.append(f)
        if verbose:
            print(f"  fold{k} val F1 {f['val_macro_f1']:.4f} train F1 {f['train_macro_f1']:.4f} top1 {f['val_top1']:.4f}", flush=True)
    fd = pd.DataFrame(folds)
    summary = dict(cv_macro_f1_mean=fd.val_macro_f1.mean(), cv_macro_f1_std=fd.val_macro_f1.std(ddof=0),
                   cv_top1_mean=fd.val_top1.mean(), cv_logloss_mean=fd.val_logloss.mean(),
                   cv_train_macro_f1_mean=fd.train_macro_f1.mean(), cv_train_logloss_mean=fd.train_logloss.mean())
    return summary, folds


def final_eval(make, tr, te, meta_te, arcs_te):
    """Fit on all training rows, score the student-test trips. Returns (row dict, probs, top table, confusion, model)."""
    m = make()
    t0 = time.perf_counter()
    m.fit(tr[FC], tr.y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = m.predict_proba(te[FC])[:, 1]
    inf_s = time.perf_counter() - t0
    n_obs = int(meta_te.n_obs.sum())
    top1, cont, top = obs_metrics(te, p, arcs_te, n_obs_total=n_obs)
    cm = cls_metrics(te.y.values, p)
    f1_dec, yhat1 = top1_decision_f1(te, p)
    miss = float((meta_te.miss_rate * meta_te.n_obs).sum() / n_obs)
    top = top.copy()
    top["ok"] = top.arc == top.ref_arc
    per_trip = top.groupby("trip").ok.mean()
    row = dict(**cm, top1_decision_macro_f1=f1_dec, obs_top1=top1, route_continuity=cont, candidate_miss_rate=miss,
               train_time_s=fit_s, inference_time_s=inf_s, inference_ms_per_obs=1000 * inf_s / n_obs,
               top1_min_trip=per_trip.min(), top1_max_trip=per_trip.max())
    return row, p, top, confusion(te.y, yhat1), m


def save_scores(te, p, name):
    """trip, obs, arc, prob  -> input of the HMM."""
    path = os.path.join(RESULTS, f"scores_student_test_{name}.csv")
    te[["trip", "obs", "arc"]].assign(prob=p).to_csv(path, index=False)
    return path


def plot_explanation(name, m):
    """Feature importance (tree models) or coefficients (linear models). Never allowed to kill a long run."""
    try:
        if name in ("RandomForest", "XGBoost"):
            s = pd.Series(m.feature_importances_, index=FC).sort_values()
            title, color = f"{name} feature importance", "#2a6f97"
        elif name in ("LogisticRegression", "SVM_LinearCalibrated"):
            names = m[0].get_feature_names_out(FC)                  # features + missing-value indicators
            if name == "LogisticRegression":
                coef = m[-1].coef_[0]
            else:
                coef = np.mean([c.estimator.coef_[0] for c in m[-1].calibrated_classifiers_], axis=0)
            s = pd.Series(coef, index=names).sort_values()
            title, color = f"{name} standardised coefficients", "#bc4749"
        else:
            return
        plt.figure(figsize=(7, 4))
        s.plot.barh(color=color)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES, f"{name}_explanation.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("  (explanation plot skipped:", e, ")")


def run_suite(models, prefix):
    """models: {name: dict(family=str, make=callable)}. Writes results/<prefix>_*.csv and figures."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    np.random.seed(SEED)
    max_obs = MAX_OBS_TRAIN
    tr = load_train(max_obs)
    te, meta_te, arcs_te = load_test()

    cv_rows, final_rows, fold_rows, configs, cms, per_trip_rows = [], [], [], {}, {}, []
    for name, spec in models.items():
        print("=== ", name, flush=True)
        make = spec["make"]
        configs[name] = {k: str(v) for k, v in make().get_params(deep=False).items()}
        summary, folds = cv_eval(make, tr)
        cv_rows.append(dict(model=name, **summary))
        fold_rows += [dict(model=name, **f) for f in folds]

        row, p, top, cm, m = final_eval(make, tr, te, meta_te, arcs_te)
        row = dict(model=name, family=spec["family"], **row)
        final_rows.append(row)
        cms[name] = cm
        per_trip_rows += [dict(model=name, trip=t, top1=v) for t, v in top.groupby("trip").ok.mean().items()]
        print("  TEST", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        joblib.dump(dict(model=m, features=FC, seed=SEED, name=name), os.path.join(MODELS_DIR, f"{name}.joblib"))
        save_scores(te, p, name)

        plot_explanation(name, m)

    cv_df, fin = pd.DataFrame(cv_rows), pd.DataFrame(final_rows)
    cv_df.to_csv(os.path.join(RESULTS, f"{prefix}_cv_results.csv"), index=False)
    fin.to_csv(os.path.join(RESULTS, f"{prefix}_student_test_results.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RESULTS, f"{prefix}_cv_fold_detail.csv"), index=False)
    pd.DataFrame(per_trip_rows).to_csv(os.path.join(RESULTS, f"{prefix}_per_trip_top1.csv"), index=False)
    json.dump(dict(seed=SEED, features=FC, configs=configs, group_cv="GroupKFold(5) on trip id",
                   max_obs_per_train_trip=max_obs, candidate_radius_m=100),
              open(os.path.join(RESULTS, f"{prefix}_configs.json"), "w"), indent=2)

    # ---- figures (real results only)
    names = list(models)
    cols = plt.cm.tab10(np.arange(len(names)))
    fd = pd.DataFrame(fold_rows)
    plt.figure(figsize=(1.6 * len(names) + 3, 4))
    plt.boxplot([fd[fd.model == n].val_macro_f1 for n in names], tick_labels=names)
    plt.ylabel("CV macro F1 (5 folds, group-aware)")
    plt.title("Cross-validation macro F1")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{prefix}_cv_macro_f1.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a_, c, t in zip(ax, ["obs_top1", "macro_f1", "log_loss"],
                        ["Student-test Top-1", "Student-test macro F1 (thr 0.5)", "Student-test log loss (lower better)"]):
        a_.bar(names, fin[c], color=cols)
        a_.set_title(t)
        a_.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{prefix}_student_test_comparison.png"), dpi=150)
    plt.close()

    x, w = np.arange(len(names)), .38
    plt.figure(figsize=(1.6 * len(names) + 3, 4))
    plt.bar(x - w / 2, cv_df.cv_train_macro_f1_mean, w, label="train (in-fold)")
    plt.bar(x + w / 2, cv_df.cv_macro_f1_mean, w, label="validation (held-out trips)")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Macro F1")
    plt.legend()
    plt.title("Over/underfitting check")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{prefix}_overfit_check.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, len(names), figsize=(5 * len(names), 3.6), squeeze=False)
    for a_, n in zip(ax[0], names):
        ConfusionMatrixDisplay(cms[n], display_labels=["not ref arc", "ref arc"]).plot(ax=a_, colorbar=False, values_format="d")
        a_.set_title(n + "\n(top-1 decision per obs)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{prefix}_confusion_matrices.png"), dpi=150)
    plt.close()

    print(cv_df.round(4).to_string())
    print(fin.round(4).T.to_string())
    return cv_df, fin
