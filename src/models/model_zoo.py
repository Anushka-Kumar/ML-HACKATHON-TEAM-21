"""All five classifiers in one place, so the comparison, Bayesian optimisation and the API build
them the same way.

  1 RandomForest          (bagged trees)
  2 KNN                   (instance based)
  3 LogisticRegression    (linear, probabilistic)
  4 XGBoost               (gradient boosted trees)
  5 SVM_LinearCalibrated  (linear SVM + probability calibration)

Missing values: 4 of the 8 features can be missing (heading difference when the vehicle is stationary, speed,
acceleration and turn angle at the start of a trip or after a time gap).
  * RandomForest and XGBoost handle NaN natively (the split search sends missing values to the better side).
  * KNN, LogisticRegression and SVM cannot, so they start with SimpleImputer(median, add_indicator=True):
    missing values are replaced by the training median AND an extra 0/1 column tells the model "this was missing".

Why model 5 is a *linear* SVM and not an RBF SVC (documented deviation):
  A kernel SVC trains in roughly O(n^2)-O(n^3) time. Our training table has ~365k candidate rows even after
  subsampling, and predicting needs a kernel evaluation against every support vector for each test row.
  That is hours, not minutes. LinearSVC is the same max-margin idea with a linear boundary and trains in seconds.
  SVMs give scores, not probabilities, and the HMM needs probabilities, so we wrap it in
  CalibratedClassifierCV (sigmoid / Platt scaling).
"""
import os
import sys

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_engineering"))
from data import SEED  # noqa: E402

FAMILY = {
    "RandomForest": "Ensemble (bagged trees)",
    "KNN": "Instance-based",
    "LogisticRegression": "Linear / probabilistic",
    "XGBoost": "Ensemble (gradient boosted trees)",
    "SVM_LinearCalibrated": "Max-margin (linear SVM + calibration)",
}

DEFAULTS = {
    "RandomForest": dict(n_estimators=200, max_depth=16, min_samples_leaf=5, max_features="sqrt"),
    "KNN": dict(n_neighbors=25, weights="distance", p=2),
    "LogisticRegression": dict(C=1.0),
    "XGBoost": dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0),
    "SVM_LinearCalibrated": dict(C=1.0),
}


def _prep():
    """Imputer (+ missing indicators) then scaler, for the models that need complete, comparable-scale inputs."""
    return [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]


def make_model(name, **overrides):
    """Build a fresh, unfitted model. `overrides` replace DEFAULTS (used by Bayesian optimisation)."""
    p = {**DEFAULTS[name], **overrides}
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                      n_jobs=-1, random_state=SEED)
    if name == "KNN":
        return make_pipeline(*_prep(), KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"],
                                                            p=p["p"], n_jobs=-1))
    if name == "LogisticRegression":
        return make_pipeline(*_prep(), LogisticRegression(C=p["C"], max_iter=2000, random_state=SEED))
    if name == "XGBoost":
        # tree_method="hist": fast histogram split finding. eval_metric only labels the loss; the objective is
        # binary logistic loss. No early stopping (we do group-aware CV ourselves).
        return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                             max_depth=p["max_depth"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                             reg_lambda=p["reg_lambda"], tree_method="hist", objective="binary:logistic",
                             eval_metric="logloss", n_jobs=-1, random_state=SEED)
    if name == "SVM_LinearCalibrated":
        # dual=False is the faster solver when rows >> features
        base = LinearSVC(C=p["C"], dual=False, max_iter=5000, random_state=SEED)
        return make_pipeline(*_prep(), CalibratedClassifierCV(base, method="sigmoid", cv=3))
    raise ValueError(f"unknown model {name}")


def factory(name, **overrides):
    """Zero-argument function returning a new model (what the CV loop wants)."""
    return lambda: make_model(name, **overrides)


def describe(name, params=None):
    """Short human readable key configuration for the comparison table."""
    p = {**DEFAULTS[name], **(params or {})}
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())
