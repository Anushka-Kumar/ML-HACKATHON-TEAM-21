# Model documentation

Rule for this file: every result is written from the CSV/log produced by the code. Placeholders look like
`[[PASTE: file -> column]]` and must be replaced by the real value from that file. Initial hyperparameters are
facts from `src/models/model_zoo.py` and can be quoted as they are.

## Shared setup (applies to all five models)
- **Task:** for every GPS observation we have 5 candidate road arcs (the 5 nearest by point-to-segment distance,
  found among the 100 nearest arc midpoints; Person 1's `improved_noise_features.py`). The classifier answers
  "is this candidate the arc the vehicle is on?" (binary label, 1 = reference arc). The candidate with the highest
  probability per observation is the prediction (Top-1).
- **Features (8, team CSV):** distance_to_arc_m, heading_difference_deg, candidate_rank, gps_speed_mps,
  gps_acceleration_mps2, gps_turn_angle_deg, road_segment_length_m, network_connectivity (definitions:
  `src/feature_engineering/phase2_data.py`). The feature files carry no labels; labels (reference arc per observation,
  from aligning the cleaned GPS track to the reference route) live in `labels_<part>.parquet` and are joined only
  inside the training/evaluation code.
- **Missing values:** heading difference, speed, acceleration and turn angle can be missing (stationary vehicle,
  trip start, time gaps). Random Forest and XGBoost handle NaN natively; KNN, Logistic Regression and the SVM use a
  median imputer plus missing-value indicator columns.
- **GPS noise handling:** isolated spikes are flagged and interpolated by Person 1's screening before the features are
  computed.
- **Validation:** GroupKFold(5) by trip id on the 80 training trips (no trip appears in both train and validation),
  then final fit on all 80 and evaluation on the 10 student-test trips. Seed 42.
- **Training-only label-noise filter:** observations whose reference arc is not among the candidates are dropped from
  training. Test observations are never dropped; they count as wrong in Top-1.
- **Known data caveat:** reference routes are not perfect ground truth. Trips 54, 33, 2 and 66 have reference routes
  more than 100 m from the GPS points, and about 99% of candidate misses come from these four trips.
  Candidate miss rate (reference arc not among the 5 candidates), student-test:
  `[[PASTE: results/p2_student_test_results.csv -> candidate_miss_rate]]` (an upper limit on Top-1).
- **Metrics:** candidate level (threshold 0.5) accuracy, precision, recall, macro F1, log loss, confusion matrix;
  task level Top-1, candidate miss rate, route continuity, training time, inference time.
  Definitions: `src/evaluation/metrics.py`.

Template used for each model below: why suitable / strengths and limits / hyperparameters / loss / results / fitting.

---
## 1. Random Forest
- **Why suitable:** non-linear interactions between distance, heading and speed; no feature scaling needed; robust to
  noisy labels.
- **Strengths:** handles mixed-scale features, gives feature importances, rarely overfits badly because trees are averaged.
- **Limits:** probabilities from tree votes can be poorly calibrated; large model, slower inference than linear models.
- **Initial hyperparameters:** n_estimators=200, max_depth=16, min_samples_leaf=5, max_features="sqrt", seed 42.
- **Loss/objective:** each tree minimises Gini impurity; the forest averages tree probabilities.
- **Group-CV:** macro F1 `[[PASTE: p2_cv_results.csv -> cv_macro_f1_mean ± cv_macro_f1_std]]`, Top-1 `[[PASTE: cv_top1_mean]]`, log loss `[[PASTE: cv_logloss_mean]]`
- **Student-test:** Top-1 `[[PASTE: p2_student_test_results.csv -> obs_top1]]`, macro F1 `[[PASTE: macro_f1]]`, log loss `[[PASTE: log_loss]]`, route continuity `[[PASTE: route_continuity]]`, inference ms/obs `[[PASTE: inference_ms_per_obs]]`
- **Over/underfitting evidence:** train macro F1 `[[PASTE: cv_train_macro_f1_mean]]` vs validation `[[PASTE: cv_macro_f1_mean]]` (figure `figures/overfit_check.png`). Interpretation: `[[WRITE after reading the gap]]`

## 2. K-Nearest Neighbours
- **Why suitable:** a simple instance-based baseline; candidates with similar feature vectors tend to share a label.
- **Strengths:** no training, no assumptions about the decision boundary.
- **Limits:** needs scaled features; inference cost grows with the training set; sensitive to irrelevant features;
  with distance weighting the training score is perfect (each point is its own nearest neighbour), so the in-fold
  training score is not a useful quality measure.
- **Initial hyperparameters:** n_neighbors=25, weights="distance", p=2 (Euclidean), StandardScaler first.
- **Loss/objective:** none (lazy learner); probability = distance-weighted share of positive neighbours.
- **Group-CV / Student-test / Over-underfitting:** same fields as model 1, taken from the same CSV rows (`model == KNN`).
  `[[PASTE]]`

## 3. Logistic Regression
- **Why suitable:** transparent linear baseline that outputs real probabilities.
- **Strengths:** fast, calibrated by construction, coefficients explain the effect of each feature (`figures/lr_coefficients.png`).
- **Limits:** linear in the features, so it cannot model interactions unless features encode them (we add cos_dh and speed_x_cos).
- **Initial hyperparameters:** C=1.0 (L2 penalty), max_iter=2000, StandardScaler first.
- **Loss/objective:** minimise log loss + L2 penalty (1/C weighting).
- **Results:** `[[PASTE]]` (rows `model == LogisticRegression`).

## 4. XGBoost
- **Why suitable:** boosting captures complex non-linear patterns with small trees added sequentially; XGBoost adds
  regularisation, row/column subsampling and native handling of missing values (useful for our NaN features).
- **Strengths:** usually the strongest on tabular data; handles feature interactions; fast inference.
- **Limits:** many hyperparameters; can overfit with deep trees / many rounds; early stopping is off so the number of
  rounds is fixed (our validation is group-aware CV, not a random split of rows).
- **Initial hyperparameters:** n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
  min_child_weight=1, reg_lambda=1.0, tree_method="hist", seed 42.
- **Loss/objective:** binary logistic loss + L2 penalty on leaf weights, optimised by second-order gradient boosting.
- **Results:** `[[PASTE: p3_*.csv rows model == XGBoost]]`

## 5. SVM (linear, calibrated)
- **Why suitable:** max-margin classifier; robust with many correlated features.
- **Deviation from a textbook RBF SVM (documented):** kernel SVC is O(n²–n³) in the number of rows. With about 365k
  training rows and 531k test rows it would take hours. We use LinearSVC (same max-margin idea, linear boundary) and
  convert its margins to probabilities with Platt scaling (CalibratedClassifierCV, sigmoid, cv=3) because the HMM
  needs probabilities. Calibration folds inside the training data are not grouped by trip, so calibration can be
  slightly optimistic; the outer group-CV and student-test scores are unaffected.
- **Strengths:** fast, convex, stable. **Limits:** linear boundary only; probabilities exist only through calibration.
- **Initial hyperparameters:** C=1.0, squared hinge loss, dual=False, StandardScaler first.
- **Loss/objective:** minimise ½‖w‖² + C·Σ squared-hinge(margin).
- **Results:** `[[PASTE: p3_*.csv rows model == SVM_LinearCalibrated]]`

---
## Model comparison
Table: `results/model_comparison.csv`. Selection rule (fixed in advance): highest group-CV Top-1; models within 0.002
are tie-broken by lowest CV log loss. Chosen model: `[[PASTE: results/final_model.json -> name]]`.
Why that model won: `[[WRITE from the table]]`.

## Bayesian optimisation
- Optuna TPE, objective = mean group-CV Top-1 on the 80 training trips, trial 0 = default configuration.
- Search space: `src/models/bayes_opt.py -> suggest()`. Every trial: `results/bo_trials_<model>.csv`.
- Best parameters: `[[PASTE: results/bo_best_params_<model>.json]]`
- Tuned vs untuned on student-test: `[[PASTE: results/bo_tuned_vs_untuned_<model>.csv]]`

## HMM / Viterbi
- Hidden state = candidate arc; emission = classifier probability^emission_power; transition from road connectivity
  (hops along the road direction, penalty for disconnected jumps, reset after long GPS gaps). Code: `src/hmm/viterbi_hmm.py`.
- Decoder parameters tuned on out-of-fold scores of training trips only. Chosen: `[[PASTE: results/hmm_best_params.json]]`
- Classifier-only vs classifier + HMM on student-test: `[[PASTE: results/hmm_comparison.csv]]`

## Honest limitations
- Reference routes are derived labels, not surveyed truth.
- Top-1 cannot exceed `1 - candidate_miss_rate`. `[[PASTE miss rate]]`
- Student-test has 10 trips; per-trip spread: `[[PASTE: top1_min_trip, top1_max_trip]]`.
