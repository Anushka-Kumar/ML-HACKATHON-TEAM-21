
import os
import numpy as np
import pandas as pd

TRAIN_DIR = "data/train"
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mapping_rows = []

trip_folders = sorted(
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
)

for trip_id in trip_folders:
    trip_dir = os.path.join(TRAIN_DIR, trip_id)

    nodes_path = os.path.join(trip_dir, trip_id + ".nodes")
    arcs_path = os.path.join(trip_dir, trip_id + ".arcs")
    route_path = os.path.join(trip_dir, trip_id + ".route")

    if not all(os.path.exists(p) for p in [
        nodes_path, arcs_path, route_path
    ]):
        print("Skipping incomplete trip:", trip_id)
        continue

    nodes = np.loadtxt(nodes_path, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)
    route = np.loadtxt(
        route_path, dtype=int, ndmin=2
    ).flatten()

    route_arc_ids = set(route.tolist())

    for candidate_arc_id, (from_node, to_node) in enumerate(arcs):

        if not (
            0 <= from_node < len(nodes)
            and 0 <= to_node < len(nodes)
        ):
            continue

        start = nodes[from_node]
        end = nodes[to_node]

        mapping_rows.append({
            "trip_id": trip_id,
            "candidate_arc_id": candidate_arc_id,
            "from_node_id": int(from_node),
            "to_node_id": int(to_node),
            "from_longitude": float(start[0]),
            "from_latitude": float(start[1]),
            "to_longitude": float(end[0]),
            "to_latitude": float(end[1]),
            "is_in_reference_route": (
                candidate_arc_id in route_arc_ids
            )
        })

mapping = pd.DataFrame(mapping_rows)

output_path = os.path.join(
    OUTPUT_DIR,
    "road_segment_mapping.csv"
)

mapping.to_csv(output_path, index=False)

print("\nROAD MAPPING GENERATED")
print("Training trips:", len(trip_folders))
print("Mapped road segments:", len(mapping))
print("Unique trips:", mapping["trip_id"].nunique())
print("Output:", output_path)
print("\nFirst five rows:")
print(mapping.head().to_string(index=False))
