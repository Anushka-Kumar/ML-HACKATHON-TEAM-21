"""Bayesian hyperparameter optimisation (Optuna, TPE sampler) of the model chosen by compare_models.py.

Run:  python src\\models\\bayes_opt.py *> results\\bo_run.log
Optional environment variables:
  BO_MODEL      model name (default: the one in results/final_model.json)
  BO_TRIALS     number of trials (default 25)
  BO_MAX_OBS    observations per training trip used INSIDE the search (default 300, smaller = faster)

How it works
  * Objective (to MAXIMISE) = mean Top-1 accuracy over GroupKFold(5) on the 80 TRAINING trips.
    Group-aware, so the score is about unseen trips. Student-test and evaluator trips are never touched while searching.
  * TPE (Tree-structured Parzen Estimator) models "good" vs "bad" parameter regions from the trials so far and
    proposes the next trial where good settings are likely -- smarter than a blind grid or random search.
  * Trial 0 is the DEFAULT configuration, so the baseline is scored with exactly the same data and folds.
  * Every trial is written to results/bo_trials_<model>.csv.
  * Afterwards: refit the best parameters on the full training set (same size as the untuned model) and compare
    tuned vs untuned on the student-test trips. The variant that is carried forward is chosen by the CV objective
    (tuned only if its CV Top-1 beats the default's), never by the test score.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.dirname(__file__))
from metrics import FC, SEED, RESULTS, FIGURES, MODELS_DIR, load_train, load_test  # noqa: E402
from model_zoo import DEFAULTS, factory  # noqa: E402
from suite import cv_eval, final_eval, save_scores  # noqa: E402


def suggest(trial, name):
    """Search space per model (ranges chosen around the defaults, wide enough to find under/over-fit regimes)."""
    if name == "RandomForest":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                    max_depth=trial.suggest_int("max_depth", 6, 30),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
    if name == "KNN":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 5, 100),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    p=trial.suggest_categorical("p", [1, 2]))
    if name == "LogisticRegression":
        return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True))
    if name == "XGBoost":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                    learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True))
    if name == "SVM_LinearCalibrated":
        return dict(C=trial.suggest_float("C", 1e-3, 10.0, log=True))
    raise ValueError(name)


def main():
    fm = json.load(open(os.path.join(RESULTS, "final_model.json")))
    name = os.environ.get("BO_MODEL", fm["name"])
    n_trials = int(os.environ.get("BO_TRIALS", 25))
    bo_obs = int(os.environ.get("BO_MAX_OBS", 300))
    print(f"Optimising {name}: {n_trials} trials, {bo_obs} obs/train-trip inside the search")

    tr_search = load_train(bo_obs)
    trials_csv = os.path.join(RESULTS, f"bo_trials_{name}.csv")

    def objective(trial):
        params = suggest(trial, name)
        summary, _ = cv_eval(factory(name, **params), tr_search, verbose=False)
        trial.set_user_attr("cv_macro_f1", summary["cv_macro_f1_mean"])
        trial.set_user_attr("cv_logloss", summary["cv_logloss_mean"])
        return summary["cv_top1_mean"]

    def log_trials(study, _trial):                       # rewrite the CSV after every trial (survives a crash)
        study.trials_dataframe().to_csv(trials_csv, index=False)
        t = study.trials[-1]
        print(f"  trial {t.number:>3}  CV Top-1 {t.value:.4f}  best so far {study.best_value:.4f}  {t.params}", flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    default_in_space = {k: v for k, v in DEFAULTS[name].items()}
    study.enqueue_trial(default_in_space)               # trial 0 = default configuration (baseline)
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trials])

    default_cv = study.trials[0].value
    best_cv, best_params = study.best_value, study.best_params
    use_tuned = best_cv > default_cv
    print(f"\nDefault CV Top-1 {default_cv:.4f} | best {best_cv:.4f} | params {best_params}")
    json.dump(dict(model=name, best_params=best_params, best_cv_top1=best_cv, default_cv_top1=default_cv,
                   n_trials=n_trials, search_obs_per_trip=bo_obs, seed=SEED),
              open(os.path.join(RESULTS, f"bo_best_params_{name}.json"), "w"), indent=2)

    # ---- refit on the same training set the untuned model used, compare on the student-test trips
    tr = load_train()
    te, meta_te, arcs_te = load_test()
    row_t, p_t, _, _, m_t = final_eval(factory(name, **best_params), tr, te, meta_te, arcs_te)
    row_u, p_u, _, _, m_u = final_eval(factory(name), tr, te, meta_te, arcs_te)   # same data, defaults
    comp = pd.DataFrame([dict(model=name, variant="untuned", **row_u), dict(model=name, variant="tuned", **row_t)])
    comp.to_csv(os.path.join(RESULTS, f"bo_tuned_vs_untuned_{name}.csv"), index=False)
    print(comp.round(4).T.to_string())

    joblib.dump(dict(model=m_t, features=FC, seed=SEED, name=name, params=best_params),
                os.path.join(MODELS_DIR, f"{name}_tuned.joblib"))
    save_scores(te, p_t, f"{name}_tuned")

    # model carried forward (chosen by CV objective only)
    if use_tuned:
        final = dict(name=name, variant="tuned", params=best_params, model_path=f"models/{name}_tuned.joblib",
                     scores_path=f"results/scores_student_test_{name}_tuned.csv")
    else:
        final = dict(name=name, variant="untuned", params={}, model_path=f"models/{name}.joblib",
                     scores_path=f"results/scores_student_test_{name}.csv")
    final.update(rule="tuned only if CV Top-1 of best trial > CV Top-1 of default trial", cv_top1=max(best_cv, default_cv))
    json.dump(final, open(os.path.join(RESULTS, "final_model.json"), "w"), indent=2)
    print("Carried forward:", final["variant"])

    # ---- optimisation history plot
    vals = [t.value for t in study.trials]
    best_so_far = pd.Series(vals).cummax()
    plt.figure(figsize=(6, 4))
    plt.plot(vals, "o", alpha=.6, label="trial")
    plt.plot(best_so_far, "-", label="best so far")
    plt.axhline(default_cv, color="gray", ls="--", label="default")
    plt.xlabel("trial")
    plt.ylabel("CV Top-1")
    plt.title(f"Bayesian optimisation: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"bo_history_{name}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
