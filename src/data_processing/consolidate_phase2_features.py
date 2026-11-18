
from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "phase2"
OUTPUT_DIR = INPUT_DIR

OUTPUT_FILE = OUTPUT_DIR / "all_trips_expanded_candidate_features.csv"
QUALITY_REPORT = OUTPUT_DIR / "phase2_feature_quality_report.csv"
OBSERVATION_REPORT = OUTPUT_DIR / "phase2_observation_candidate_counts.csv"

FEATURE_COLUMNS = [
    "distance_to_arc_m",
    "heading_difference_deg",
    "candidate_rank",
    "gps_speed_mps",
    "gps_acceleration_mps2",
    "gps_turn_angle_deg",
    "road_segment_length_m",
    "network_connectivity",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*_expanded_candidate_features.csv"))

    if not files:
        print("ERROR: No per-trip feature CSV files found.")
        print(f"Expected files in: {INPUT_DIR}")
        return

    print(f"Found {len(files)} per-trip feature files.")

    dataframes = []

    for file in files:
        try:
            df = pd.read_csv(file)
            if "trip_id" not in df.columns:
                trip_id = file.name.split("_")[0]
                df["trip_id"] = trip_id

            dataframes.append(df)
            print(f"Loaded: {file.name} ({len(df)} rows)")

        except Exception as error:
            print(f"ERROR reading {file.name}: {error}")

    if not dataframes:
        print("ERROR: No files could be loaded.")
        return

    combined = pd.concat(dataframes, ignore_index=True)

    # Check required feature columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in combined.columns
    ]

    if missing_columns:
        print("ERROR: Required feature columns are missing:")
        print(missing_columns)
        return

    # Save combined feature dataset
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nCombined dataset saved.")
    print(f"Total rows: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")
    print(f"Output: {OUTPUT_FILE}")

    # Feature quality report
    quality_rows = []

    for column in FEATURE_COLUMNS:
        numeric = pd.to_numeric(combined[column], errors="coerce")

        missing_count = int(numeric.isna().sum())
        infinite_count = int(np.isinf(numeric.dropna()).sum())

        valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()

        quality_rows.append({
            "feature": column,
            "missing_values": missing_count,
            "infinite_values": infinite_count,
            "minimum": valid.min() if len(valid) else np.nan,
            "median": valid.median() if len(valid) else np.nan,
            "p95": valid.quantile(0.95) if len(valid) else np.nan,
            "maximum": valid.max() if len(valid) else np.nan,
        })

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(QUALITY_REPORT, index=False)

    # Candidate count per observation
    group_columns = ["trip_id", "observation_id"]

    observation_df = (
        combined.groupby(group_columns)
        .size()
        .reset_index(name="candidate_count")
    )

    observation_df["expected_candidate_count"] = 5
    observation_df["matches_expected"] = (
        observation_df["candidate_count"] == 5
    )

    observation_df.to_csv(OBSERVATION_REPORT, index=False)

    # Duplicate check
    duplicate_columns = [
        col for col in
        ["trip_id", "observation_id", "candidate_rank"]
        if col in combined.columns
    ]

    duplicate_count = int(
        combined.duplicated(subset=duplicate_columns).sum()
    ) if duplicate_columns else 0

    print("\n========== PHASE 2 CONSOLIDATION ==========")
    print(f"Trips processed: {len(dataframes)}")
    print(f"Total candidate rows: {len(combined)}")
    print(f"Unique observations: {len(observation_df)}")
    print(f"Duplicate candidate rows: {duplicate_count}")
    print(
        "Observations with candidate count != 5: "
        f"{int((~observation_df['matches_expected']).sum())}"
    )

    print("\nFeature quality report:")
    print(quality_df.to_string(index=False))

    print("\nGenerated files:")
    print(OUTPUT_FILE)
    print(QUALITY_REPORT)
    print(OBSERVATION_REPORT)

    print("\nConsolidation completed successfully.")


if __name__ == "__main__":
    main()
