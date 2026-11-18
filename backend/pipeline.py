"""Full map-matching pipeline for ONE trip, used by the API.

    trip id -> load track + road network -> candidate arcs + features -> classifier probabilities
            -> HMM/Viterbi -> predicted route -> (server side) evaluation against the reference route

Ground-truth rule: steps 1-4 only ever see the .track, .nodes and .arcs files. The .route file (reference) is read
in `evaluate()` AFTER the prediction is finished, and only to compute metrics; it never reaches the model or HMM.
"""
import json
import logging
import os
import sys
import time

import joblib
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("feature_engineering", "hmm"):
    sys.path.insert(0, os.path.join(REPO, "src", sub))
import data as D  # noqa: E402                    (feature_engineering/data.py)
from phase2_data import compute_features  # noqa: E402   (8-feature candidate table from raw files)
from viterbi_hmm import (DEFAULT_PARAMS, TripHMM, classifier_only, score_prediction,  # noqa: E402
                         route_from_prediction)

log = logging.getLogger("pathsnap")
EVAL_PART = os.environ.get("EVAL_PART", "student_test")        # folder under PATHSNAP_DATA holding the trips to serve
SHOW_REFERENCE = os.environ.get("SHOW_REFERENCE", "1") == "1"  # return reference geometry to the UI after prediction


class Pipeline:
    def __init__(self):
        fm_path = os.path.join(REPO, "results", "final_model.json")
        fm = json.load(open(fm_path)) if os.path.exists(fm_path) else None
        model_path = os.path.join(REPO, fm["model_path"]) if fm else None
        self.model = None
        self.features = None
        self.model_name = "Model not configured"
        self.model_error = None
        if model_path and os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.features = bundle["features"]
                self.model_name = f"{fm['name']} ({fm['variant']})"
            except (OSError, KeyError, ValueError) as error:
                self.model_error = str(error)
        elif model_path:
            self.model_error = f"Model bundle not found: {model_path}"
        else:
            self.model_error = "results/final_model.json not found"
        hp = os.path.join(REPO, "results", "hmm_best_params.json")
        self.hmm_params = json.load(open(hp)) if os.path.exists(hp) else DEFAULT_PARAMS
        log.info("loaded model %s | HMM params %s | trips folder %s/%s", self.model_name, self.hmm_params, D.ROOT, EVAL_PART)

    def trips(self):
        return D.trips(EVAL_PART)

    # ------------------------------------------------------------------ prediction (no ground truth)
    def predict(self, trip_id):
        if self.model is None:
            raise RuntimeError(
                "Production model is not available. "
                f"{self.model_error or 'Configure the model bundle.'}"
            )
        t0 = time.perf_counter()
        base = f"{D.ROOT}/{EVAL_PART}/{trip_id}/{trip_id}"
        track = np.loadtxt(base + ".track")
        nodes = np.loadtxt(base + ".nodes")
        arcs = np.loadtxt(base + ".arcs", dtype=int)
        log.info("[%s] loaded %d GPS points, %d nodes, %d arcs", trip_id, len(track), len(nodes), len(arcs))

        df, aux = compute_features(track, nodes, arcs)               # 5 candidates + 8 features, NO labels
        t1 = time.perf_counter()
        log.info("[%s] %d candidate rows (%.1f s)", trip_id, len(df), t1 - t0)

        df["prob"] = self.model.predict_proba(df[self.features])[:, 1]
        t2 = time.perf_counter()
        log.info("[%s] classifier scored candidates (%.1f s)", trip_id, t2 - t1)

        clf = classifier_only(df)
        hmm = TripHMM(arcs, df).decode(self.hmm_params)
        t3 = time.perf_counter()
        log.info("[%s] HMM/Viterbi decoded %d observations (%.1f s)", trip_id, len(hmm), t3 - t2)
        return dict(track=track, nodes=nodes, arcs=arcs, df=df, aux=aux, clf=clf, hmm=hmm,
                    timing=dict(features_s=t1 - t0, classifier_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0))

    # ------------------------------------------------------------------ server-side evaluation
    def evaluate(self, trip_id, pr):
        """Compare predictions with the reference route. Returns (metrics dict | None, reference route | None)."""
        route_path = f"{D.ROOT}/{EVAL_PART}/{trip_id}/{trip_id}.route"
        if not os.path.exists(route_path):
            log.info("[%s] no reference route available -> no evaluation", trip_id)
            return None, None
        route = np.loadtxt(route_path, dtype=int, ndmin=1)
        aux = pr["aux"]                                             # cleaned GPS + arc ends, same UTM projection as training
        ref = D.reference_arc_per_obs(aux["P"], aux["a"], aux["b"], route)   # reference arc per GPS observation
        n = len(pr["track"])
        out = {}
        for key in ("clf", "hmm"):
            m = score_prediction(pr[key], ref, n, pr["arcs"], route)
            out[key] = dict(top1=m["top1"], route_continuity=m["cont_ok"] / max(m["cont_n"], 1),
                            route_precision=m["route_precision"], route_recall=m["route_recall"], route_f1=m["route_f1"])
        in_cand = (pr["df"]["arc"].values == ref[pr["df"]["obs"].values])
        covered = np.zeros(n, dtype=bool)
        covered[pr["df"]["obs"].values[in_cand]] = True
        out["candidate_miss_rate"] = float(1 - covered.mean())
        log.info("[%s] Top-1 classifier %.4f -> HMM %.4f | route F1 %.4f", trip_id, out["clf"]["top1"],
                 out["hmm"]["top1"], out["hmm"]["route_f1"])
        return out, route

    # ------------------------------------------------------------------ response for the UI
    @staticmethod
    def _geom(nodes, arcs, arc_ids):
        """[[ [lat,lon], [lat,lon] ], ...] one segment per arc."""
        a = arcs[np.asarray(arc_ids, dtype=int)]
        return [[[float(nodes[f, 1]), float(nodes[f, 0])], [float(nodes[t, 1]), float(nodes[t, 0])]] for f, t in a]

    @staticmethod
    def _frontend_evaluation(m):
        """Evaluation block. Keeps our own keys (clf / hmm) and adds the names the React frontend reads."""
        if m is None:
            return dict(status="unavailable", reason="No reference route on the server for this trip.")
        return dict(status="available", **m,
                    classifier_top1=dict(pointwise_accuracy=m["clf"]["top1"], **m["clf"]),
                    viterbi=dict(pointwise_accuracy=m["hmm"]["top1"], **m["hmm"]))

    def run(self, trip_id):
        pr = self.predict(trip_id)
        metrics, route = self.evaluate(trip_id, pr)
        pred_route = route_from_prediction(pr["hmm"])
        track = pr["track"]
        stride = max(1, len(track) // 4000)                         # keep the JSON small for the map
        candidate_arcs = np.unique(pr["df"]["arc"].values)
        # per-observation view (the React frontend draws these): GPS position, the classifier's best arc and the
        # arc chosen by the HMM (None where an observation has no candidate arc)
        clf_arc = dict(zip(pr["clf"].obs.values.tolist(), pr["clf"].arc.values.tolist()))
        hmm_arc = dict(zip(pr["hmm"].obs.values.tolist(), pr["hmm"].arc.values.tolist()))
        predictions = [dict(observation_id=i, longitude=float(track[i, 0]), latitude=float(track[i, 1]),
                            timestamp=float(track[i, 2]), classifier_arc_id=clf_arc.get(i),
                            selected_arc_id=hmm_arc.get(i)) for i in range(len(track))]
        resp = dict(trip_id=trip_id, model=self.model_name, hmm_params=self.hmm_params, predictions=predictions,
                    n_gps_points=int(len(track)), timing=pr["timing"],
                    gps=[[float(y), float(x)] for x, y in track[::stride, :2]],
                    network=self._geom(pr["nodes"], pr["arcs"], candidate_arcs),
                    predicted=self._geom(pr["nodes"], pr["arcs"], pred_route), predicted_arcs=pred_route,
                    evaluation=self._frontend_evaluation(metrics))
        if SHOW_REFERENCE and route is not None:
            resp["reference"] = self._geom(pr["nodes"], pr["arcs"], np.unique(route))
        return resp
