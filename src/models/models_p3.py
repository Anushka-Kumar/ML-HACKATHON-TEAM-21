"""Models 4 and 5: XGBoost and linear SVM (calibrated).

Same shared dataset, same GroupKFold(5), same metrics module as models 1-3 (see src/evaluation/).
Run from the repo root:   python src\\models\\models_p3.py *> results\\p3_run.log

Model 4  XGBClassifier  (gradient boosting family)
         Builds trees one after another, each fixing the errors of the previous ones (second-order gradient of the
         log loss), with L2 regularisation on leaf weights and row/column subsampling. Handles NaN natively and
         captures non-linear feature interactions (distance x heading x speed).

Model 5  LinearSVC wrapped in CalibratedClassifierCV  (max-margin family)
         DEVIATION from a textbook RBF-kernel SVM: kernel SVC is O(n^2..n^3) in the number of rows and our table
         has ~365k rows (after subsampling) and 531k test rows, so it would take hours. A linear SVM is the same
         max-margin idea and trains in seconds. Raw SVM outputs are margins, not probabilities, so we calibrate with
         Platt scaling (sigmoid) -- the HMM needs probabilities.
         Calibration uses cv=3 *inside* the training data (plain stratified folds, not grouped by trip); this can make
         probabilities slightly optimistic, but the outer GroupKFold/test scores are unaffected by it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.dirname(__file__))
from model_zoo import FAMILY, factory  # noqa: E402
from suite import run_suite  # noqa: E402

MODELS = {name: dict(family=FAMILY[name], make=factory(name))
          for name in ("XGBoost", "SVM_LinearCalibrated")}

if __name__ == "__main__":
    run_suite(MODELS, prefix="p3")
