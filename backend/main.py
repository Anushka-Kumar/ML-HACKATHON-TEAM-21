"""FastAPI server (runs on Laptop 1).

Start from the repo root:
    $env:PATHSNAP_DATA = "data"
    $env:EVAL_PART = "student_test"        # sub-folder of PATHSNAP_DATA that holds the trips to serve
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            -> {"status": "ok", ...}
    GET  /trips             -> list of trip ids the UI can choose from
    POST /trips/{id}/predict -> runs the whole pipeline for that trip and evaluates server-side (React frontend)
    POST /predict           -> same, with body {"trip_id": "00000000"}
Every request is logged to the console and to logs/api.log.
"""
import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import Pipeline, EVAL_PART  # noqa: E402

os.makedirs(os.path.join(HERE, "..", "logs"), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(HERE, "..", "logs", "api.log"))])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap map-matching API")
# the UI runs on another laptop (another origin), so the browser needs CORS permission
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe = Pipeline()                       # model, HMM params and trip list are loaded once at start-up


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s -> %s (%.2f s) from %s", request.method, request.url.path, resp.status_code,
             time.perf_counter() - t0, request.client.host if request.client else "?")
    return resp


@app.get("/health")
def health():
    return dict(status="ok", model_ready=True, model=pipe.model_name, trips_folder=EVAL_PART)


@app.get("/trips")
def trips():
    return dict(trips=pipe.trips())


def _run(trip_id: str):
    # accept only ids that exist in the served folder (also blocks path tricks like "../")
    if trip_id not in pipe.trips():
        raise HTTPException(status_code=404, detail=f"unknown trip id {trip_id}")
    log.info("=== running pipeline for trip %s", trip_id)
    return pipe.run(trip_id)


@app.post("/trips/{trip_id}/predict")          # used by the React frontend (frontend/src/api.js)
def predict_trip(trip_id: str):
    return _run(trip_id)


@app.post("/predict")                          # same thing with the id in the JSON body
def predict(req: PredictRequest):
    return _run(req.trip_id)
