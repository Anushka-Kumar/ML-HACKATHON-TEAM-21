
# PathSnap – ML Hackathon Team 21

## Project Overview
PathSnap is a machine learning-based road map matching project that reconstructs road routes from GPS trajectory observations.

## Team Members

| Name | USN | Responsibility |
|---|---|---|
| Anushka P | PES1UG24CS075 | Dataset, preprocessing, feature engineering and frontend |
| Anushka K Kishore | PES1UG24CS074 | Machine Learning Models 1, 2 and 3 |
| Chaithanya N | PES1UG24CS124 | Machine Learning Models 4 and 5, comparison and Bayesian optimisation |
| Charan M | PES1UG24CS125 | HMM, Viterbi and backend |

## Project Structure

- src/data_processing – Dataset loading and preprocessing
- src/feature_engineering – Feature extraction
- src/visualization – Data visualisation
- notebooks – Exploratory analysis
- frontend – User interface
- backend – API implementation
- models – Model artifacts
- results – Experimental results

## Technologies
Python, NumPy, Pandas, Scikit-learn, Folium, NetworkX and FastAPI.


## How to run (Windows PowerShell, from the repo root)

Setup
```
py -3.13 -m venv venv          # any 3.10-3.13 works; 3.14 is too new for some wheels
.\venv\Scripts\Activate.ps1
pip install --prefer-binary -r requirements_p3.txt
$env:PATHSNAP_DATA = "data"    # data/train and data/student_test
```

Pipeline (each step writes into results/, figures/, models/)
```
# 1. team feature CSVs (Person 1's script, unchanged): one CSV per trip
python src\data_processing\improved_noise_features.py --trip-dir data\train\<trip> --output-dir results\phase2
python src\data_processing\improved_noise_features.py --trip-dir data\student_test\<trip> --output-dir results\phase2_student_test
# (run it for every trip; process_all_training.py does this for the 80 training trips)
# 2. CSVs -> parquet features + labels, candidate miss-rate report
python src\feature_engineering\phase2_data.py
python src\feature_engineering\check_feature_port.py      # API feature code == CSV code
# 3. models
python src\models\models_p2.py *> results\p2_run.log        # RF, KNN, LR
python src\models\models_p3.py *> results\p3_run.log        # XGBoost, linear SVM
python src\models\compare_models.py                         # comparison table + plots, picks the best model
python src\models\bayes_opt.py *> results\bo_run.log        # Optuna tuning of the best model
python src\models\make_oof_scores.py *> results\oof_run.log # out-of-fold scores for HMM tuning
python src\hmm\run_hmm.py *> results\hmm_run.log            # tune decoder, classifier vs classifier+HMM
python src\visualization\prediction_map.py 00000000         # Folium map of one trip
```
(`src/feature_engineering/data.py` + `build_shared_dataset.py` are the earlier 13-feature / 100 m radius pipeline;
their results are kept in `archive_v1_13feature/`.)

Deploy (two laptops)
- Laptop 1 (API), from the repo root, with models/ and results/ present:
  ```
  $env:PATHSNAP_DATA = "data"; $env:EVAL_PART = "student_test"
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```
  Allow port 8000 in Windows Firewall. Requests are logged to the console and logs\api.log.
  `EVAL_PART` is the sub-folder of PATHSNAP_DATA that holds the trips served to the UI.
- Laptop 2 (UI): the team's React frontend (`frontend/`). On Laptop 2:
  ```
  cd frontend
  npm install
  $env:VITE_API_BASE_URL = "http://<Laptop-1-IP>:8000"     # Laptop 1's address; omit on a single laptop
  npm run dev
  ```
  Open the printed URL (http://localhost:5173), pick a trip and click Reconstruct route. On a single laptop the
  Vite dev server proxies `/api` to http://127.0.0.1:8000, so no variable is needed.
  The API serves what the frontend calls: `GET /health`, `GET /trips`, `POST /trips/{id}/predict`
  (`POST /predict` with a JSON body is also accepted). `ui_simple/index.html` is a dependency-free alternative UI
  (serve with `python -m http.server 5500` inside `ui_simple`, type the API address into its box) that also draws
  the road network and the reference route as map layers.
- Ground truth stays on Laptop 1: the model and HMM never read `.route`; it is read only after prediction, to compute
  the metrics. Set `$env:SHOW_REFERENCE = "0"` to also stop the reference geometry being sent to the UI.
- models/ and results/ are in .gitignore. Either copy them to Laptop 1 or remove them from .gitignore before the final push.

Documentation: docs/MODEL_DOCUMENTATION.md (fill the placeholders from the result files).
