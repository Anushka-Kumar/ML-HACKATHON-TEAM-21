"""Models 1-3: Random Forest, KNN, Logistic Regression (on the team's 8-feature candidate dataset).

Run from the repo root:   python src/models/models_p2.py *> results/p2_run.log

Same protocol as every model (see src/evaluation/): GroupKFold(5) by trip on the 80 training trips, final fit on all
80, evaluation on the 10 student-test trips, scores saved for the HMM.
Model definitions: src/models/model_zoo.py (RF and XGBoost take NaN natively; KNN/LR/SVM impute the median first).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.dirname(__file__))
from model_zoo import FAMILY, factory  # noqa: E402
from suite import run_suite  # noqa: E402

MODELS = {name: dict(family=FAMILY[name], make=factory(name))
          for name in ("RandomForest", "KNN", "LogisticRegression")}

if __name__ == "__main__":
    run_suite(MODELS, prefix="p2")
