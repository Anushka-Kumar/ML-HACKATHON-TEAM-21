"""Folium map of one student-test trip: GPS points, road network, reference route, HMM-predicted route.

Run after src\\hmm\\run_hmm.py:
    python src\\visualization\\prediction_map.py 00000000
Output: figures/map_<trip>.html  (open in a browser)
"""
import os
import sys

import folium
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "feature_engineering"))
from data import load_trip  # noqa: E402

trip = sys.argv[1] if len(sys.argv) > 1 else "00000000"
track, nodes, arcs, route = load_trip("student_test", trip)
pred = pd.read_csv(os.path.join(REPO, "results", "hmm_predictions_student_test.csv"), dtype={"trip": str})
pred = pred[pred.trip == trip].sort_values("obs")
hmm_arcs = pred.arc_hmm.dropna().astype(int).values
hmm_route = hmm_arcs[np.r_[True, hmm_arcs[1:] != hmm_arcs[:-1]]]


def seg(a):
    f, t = arcs[a]
    return [(nodes[f, 1], nodes[f, 0]), (nodes[t, 1], nodes[t, 0])]


m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()], zoom_start=15, tiles="OpenStreetMap")
near = folium.FeatureGroup("Road network (arcs of reference + predicted route)", show=False).add_to(m)
ref_g = folium.FeatureGroup("Reference route").add_to(m)
pred_g = folium.FeatureGroup("Predicted route (HMM)").add_to(m)
gps_g = folium.FeatureGroup("GPS points").add_to(m)
for a in np.unique(np.r_[route, hmm_route]):
    folium.PolyLine(seg(a), color="#999", weight=2).add_to(near)
for a in np.unique(route):
    folium.PolyLine(seg(a), color="#2a6f97", weight=7, opacity=.6).add_to(ref_g)
for a in hmm_route:
    folium.PolyLine(seg(a), color="#d62828", weight=3).add_to(pred_g)
for lon, lat, _ in track[:: max(1, len(track) // 1500)]:
    folium.CircleMarker((lat, lon), radius=2, color="#222", fill=True, weight=0).add_to(gps_g)
folium.LayerControl().add_to(m)
legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;font-size:13px">'
          '<span style="color:#2a6f97">&#9632;</span> reference route<br><span style="color:#d62828">&#9632;</span> predicted route'
          '<br><span style="color:#222">&#9679;</span> GPS points</div>')
m.get_root().html.add_child(folium.Element(legend))
os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
out = os.path.join(REPO, "figures", f"map_{trip}.html")
m.save(out)
print("Wrote", out)
