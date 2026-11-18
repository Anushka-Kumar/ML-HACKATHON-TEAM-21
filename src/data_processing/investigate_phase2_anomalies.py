
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "phase2" / "all_trips_expanded_candidate_features.csv"
OUT = ROOT / "results" / "phase2"

df = pd.read_csv(INPUT)

print("Dataset shape:", df.shape)

# Identify extreme values
checks = {
    "gps_speed_mps": 50,
    "gps_acceleration_mps2": 10,
    "distance_to_arc_m": 500,
    "road_segment_length_m": 1000,
}

extreme_rows = []

for feature, threshold in checks.items():
    values = pd.to_numeric(df[feature], errors="coerce")
    mask = values.abs() > threshold

    print(f"\n{feature}:")
    print("Threshold:", threshold)
    print("Extreme candidate rows:", int(mask.sum()))

    selected = df.loc[mask].copy()
    selected["investigated_feature"] = feature
    selected["investigated_value"] = values[mask]

    extreme_rows.append(selected)

if extreme_rows:
    extremes = pd.concat(extreme_rows, ignore_index=True)
    extremes.to_csv(OUT / "phase2_extreme_value_rows.csv", index=False)

# Missing values by trip
missing = (
    df.groupby("trip_id")[
        ["heading_difference_deg", "gps_speed_mps",
         "gps_acceleration_mps2", "gps_turn_angle_deg"]
    ]
    .agg(lambda col: col.isna().sum())
)

missing.to_csv(OUT / "phase2_missing_values_by_trip.csv")

# Connectivity distribution
connectivity = (
    df["network_connectivity"]
    .value_counts(dropna=False)
    .rename_axis("value")
    .reset_index(name="count")
)

connectivity["percentage"] = (
    connectivity["count"] / len(df) * 100
)

connectivity.to_csv(
    OUT / "phase2_connectivity_distribution.csv",
    index=False
)

print("\nConnectivity distribution:")
print(connectivity.to_string(index=False))

print("\nGenerated reports:")
print("phase2_extreme_value_rows.csv")
print("phase2_missing_values_by_trip.csv")
print("phase2_connectivity_distribution.csv")

print("\nAnomaly investigation completed.")
