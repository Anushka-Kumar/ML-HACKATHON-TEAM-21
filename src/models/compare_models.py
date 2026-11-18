"""Merge the results of all five models into one comparison table + plots, and pick the model to carry forward.

Run after models_p2.py and models_p3.py:   python src\\models\\compare_models.py

Inputs  (all produced by code you ran): results/p2_*.csv and results/p3_*.csv
Outputs results/model_comparison.csv / .md, figures/compare_*.png, results/final_model.json

Model selection rule (decided BEFORE looking at the numbers):
  1. Highest mean group-CV Top-1 (the primary metric, measured on held-out trips).
  2. If several models are within TIE of the best, take the one with the lowest CV log loss
     (the HMM needs well-calibrated probabilities).
Student-test numbers are shown next to it for reporting but do not choose the model.
Nothing here is typed in by hand: every number is read from the result files.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model_zoo import describe  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(REPO, "results")
F = os.path.join(REPO, "figures")
TIE = 0.002          # Top-1 differences smaller than this are treated as a tie
ORDER = ["RandomForest", "KNN", "LogisticRegression", "XGBoost", "SVM_LinearCalibrated"]


def read(prefix):
    cv, te = f"{R}/{prefix}_cv_results.csv", f"{R}/{prefix}_student_test_results.csv"
    if not (os.path.exists(cv) and os.path.exists(te)):
        sys.exit(f"Missing {cv} or {te}. Run the {prefix} training script first.")
    return pd.read_csv(cv), pd.read_csv(te)


cv2, te2 = read("p2")
cv3, te3 = read("p3")
cv, te = pd.concat([cv2, cv3]), pd.concat([te2, te3])
df = cv.merge(te, on="model")
df["model"] = pd.Categorical(df["model"], ORDER, ordered=True)
df = df.sort_values("model").reset_index(drop=True)
df["key_config"] = [describe(m) for m in df["model"]]

table = df[["model", "family", "key_config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean",
            "cv_logloss_mean", "macro_f1", "obs_top1", "log_loss", "route_continuity",
            "candidate_miss_rate", "train_time_s", "inference_ms_per_obs"]].rename(columns={
    "macro_f1": "test_macro_f1", "obs_top1": "test_top1", "log_loss": "test_log_loss",
    "route_continuity": "test_route_continuity"})
table.to_csv(f"{R}/model_comparison.csv", index=False)
with open(f"{R}/model_comparison.md", "w") as fh:
    try:
        fh.write(table.round(4).to_markdown(index=False))      # needs the optional 'tabulate' package
    except ImportError:
        fh.write(table.round(4).to_string(index=False))
print(table.round(4).to_string(index=False))

# ---- model selection (CV only)
best_top1 = df.cv_top1_mean.max()
tied = df[df.cv_top1_mean >= best_top1 - TIE]
winner = tied.sort_values("cv_logloss_mean").iloc[0]
name = str(winner["model"])
final = dict(name=name, variant="untuned", params={}, model_path=f"models/{name}.joblib",
             scores_path=f"results/scores_student_test_{name}.csv",
             rule=f"highest CV Top-1; models within {TIE} of the best are tie-broken by lowest CV log loss",
             cv_top1=float(winner.cv_top1_mean), cv_logloss=float(winner.cv_logloss_mean),
             tied_models=[str(m) for m in tied["model"]])
json.dump(final, open(f"{R}/final_model.json", "w"), indent=2)
print(f"\nSelected model (by group-CV): {name}  | tied candidates: {final['tied_models']}")

# ---- plots
os.makedirs(F, exist_ok=True)
names = list(df["model"].astype(str))
x, w = np.arange(len(names)), .38
cols = plt.cm.tab10(np.arange(len(names)))


def paired(c1, c2, title, ylabel, fn, l1="group-CV (80 train trips)", l2="student-test (10 trips)"):
    plt.figure(figsize=(8, 4))
    plt.bar(x - w / 2, df[c1], w, label=l1)
    plt.bar(x + w / 2, df[c2], w, label=l2)
    plt.xticks(x, names, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{F}/{fn}", dpi=150)
    plt.close()


paired("cv_top1_mean", "obs_top1", "Observation-level Top-1", "Top-1 accuracy", "compare_top1.png")
paired("cv_macro_f1_mean", "macro_f1", "Candidate-level macro F1 (threshold 0.5)", "Macro F1", "compare_macro_f1.png")
paired("cv_logloss_mean", "log_loss", "Log loss (lower is better)", "Log loss", "compare_logloss.png")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a_, c, t, log in zip(ax, ["route_continuity", "inference_ms_per_obs", "train_time_s"],
                         ["Route continuity (student-test)", "Inference ms per observation", "Training time (s)"],
                         [False, True, True]):
    a_.bar(names, df[c], color=cols)
    a_.set_title(t)
    a_.tick_params(axis="x", rotation=25)
    if log:
        a_.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{F}/compare_continuity_speed.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4.5))
for n_, c_, (xx, yy) in zip(names, cols, zip(df.inference_ms_per_obs, df.obs_top1)):
    plt.scatter(xx, yy, color=c_, s=70)
    plt.annotate(n_, (xx, yy), textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.xscale("log")
plt.xlabel("Inference ms per observation (log)")
plt.ylabel("Student-test Top-1")
plt.title("Accuracy vs speed")
plt.tight_layout()
plt.savefig(f"{F}/compare_accuracy_vs_speed.png", dpi=150)
plt.close()
print("Wrote results/model_comparison.csv, results/final_model.json, figures/compare_*.png")
