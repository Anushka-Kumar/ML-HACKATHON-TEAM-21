# Person 2 - models on the shared candidate dataset (drop into the team repo root)

Layout (matches the team repo): src/feature_engineering/, src/models/, results/shared/, models/, figures/
Data goes in data/train and data/student_test (gitignored), or set PATHSNAP_DATA.

## Run on your laptop (from the repo root)
    pip install -r requirements_p2.txt
    # Windows PowerShell:  $env:PATHSNAP_DATA="data"      Mac/Linux: export PATHSNAP_DATA=data
    python src/feature_engineering/build_shared_dataset.py   # ~1-3 min; makes the shared candidate dataset + per-trip miss reports
    python src/models/models_p2.py                           # trains RF / KNN / LR, group CV, student-test eval
Lower training size if slow:  MAX_OBS_TRAIN=100  (default 150 observations per training trip)

## Outputs
results/shared/*  (dataset, per_trip_report_*.csv, feature_schema.csv, execution_log.txt)
results/p2_cv_results.csv, p2_student_test_results.csv, p2_cv_fold_detail.csv, p2_configs.json
results/scores_student_test_<model>.csv  -> columns trip, obs, arc, prob  (input for the HMM)
figures/*.png, models/*.joblib
