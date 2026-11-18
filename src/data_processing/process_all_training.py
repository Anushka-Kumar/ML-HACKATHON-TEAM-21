"""Run PathSnap noise screening and feature extraction for every training trip.

Expected layout:
ML-HACKATHON-TEAM-21/
    data/train/<trip_id>/
    results/phase2/
    src/data_processing/
        process_all_training.py
        improved_noise_features.py

Run from any directory:
    python process_all_training.py
"""
from pathlib import Path
import subprocess
import sys
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "results" / "phase2"
PROCESSOR = SCRIPT_DIR / "improved_noise_features.py"

REQUIRED_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def main():
    print(f"Project root: {ROOT}", flush=True)
    print(f"Training data: {TRAIN_DIR}", flush=True)
    print(f"Processor: {PROCESSOR}", flush=True)

    if not TRAIN_DIR.is_dir():
        raise SystemExit(f"Training directory not found: {TRAIN_DIR}")
    if not PROCESSOR.is_file():
        raise SystemExit(f"Missing processor: {PROCESSOR}")

    trip_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not trip_dirs:
        raise SystemExit(f"No trip folders found in {TRAIN_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    successful, failed, skipped = [], [], []

    print(f"\nFound {len(trip_dirs)} trip folders.", flush=True)
    for number, trip_dir in enumerate(trip_dirs, start=1):
        trip_id = trip_dir.name
        missing = [
            trip_id + suffix for suffix in REQUIRED_SUFFIXES
            if not (trip_dir / (trip_id + suffix)).is_file()
        ]
        if missing:
            skipped.append((trip_id, "Missing: " + ", ".join(missing)))
            print(f"[{number}/{len(trip_dirs)}] SKIP {trip_id}: {', '.join(missing)}", flush=True)
            continue

        print(f"\n[{number}/{len(trip_dirs)}] Processing {trip_id}...", flush=True)
        cmd = [
            sys.executable, str(PROCESSOR),
            "--trip-dir", str(trip_dir),
            "--output-dir", str(OUTPUT_DIR)
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode == 0:
            successful.append(trip_id)
        else:
            failed.append((trip_id, f"Exit code {result.returncode}"))
            print(f"FAILED: {trip_id}", flush=True)

    noise_files = sorted(
        f for f in OUTPUT_DIR.glob("*_noise_report.csv")
        if f.name != "all_trips_noise_report.csv"
    )
    if noise_files:
        noise = pd.concat(
            [pd.read_csv(f).assign(trip_id=f.name.replace("_noise_report.csv", ""))
             for f in noise_files],
            ignore_index=True
        )
        noise.to_csv(OUTPUT_DIR / "all_trips_noise_report.csv", index=False)

    summary_rows = []
    summary_files = sorted(
        f for f in OUTPUT_DIR.glob("*_processing_summary.csv")
        if f.name != "all_trips_processing_summary.csv"
    )
    for f in summary_files:
        try:
            s = pd.read_csv(f, index_col=0).iloc[:, 0]
            summary_rows.append({
                "trip_id": f.name.replace("_processing_summary.csv", ""),
                **s.to_dict()
            })
        except Exception as exc:
            print(f"Could not read summary {f.name}: {exc}", flush=True)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUTPUT_DIR / "all_trips_processing_summary.csv", index=False
        )

    status = (
        [{"trip_id": tid, "status": "SUCCESS", "detail": ""} for tid in successful]
        + [{"trip_id": tid, "status": "FAILED", "detail": detail} for tid, detail in failed]
        + [{"trip_id": tid, "status": "SKIPPED", "detail": detail} for tid, detail in skipped]
    )
    pd.DataFrame(status).to_csv(OUTPUT_DIR / "batch_run_status.csv", index=False)

    print("\n========== BATCH COMPLETE ==========", flush=True)
    print(f"Successful: {len(successful)}", flush=True)
    print(f"Failed:     {len(failed)}", flush=True)
    print(f"Skipped:    {len(skipped)}", flush=True)
    print(f"Outputs:    {OUTPUT_DIR}", flush=True)
    print("Original files under data/train were not modified.", flush=True)


if __name__ == "__main__":
    main()
