
import os
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from pyproj import CRS, Transformer

TRAIN_DIR = "data/train"
OUTPUT_DIR = "results/features"

# First-pass candidate-generation settings
TOP_CANDIDATES = 5
MIDPOINT_SEARCH_K = 30
HEADING_GAP_LIMIT_SECONDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_features = []

trip_folders = sorted(
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
)


def calculate_heading(x1, y1, x2, y2):
    """Calculate bearing clockwise from north, in degrees."""
    dx = x2 - x1
    dy = y2 - y1
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def angle_difference(a, b):
    """Smallest absolute angular difference."""
    return abs((a - b + 180) % 360 - 180)


for trip_id in trip_folders:
    print("Processing trip:", trip_id)

    trip_dir = os.path.join(TRAIN_DIR, trip_id)

    track_path = os.path.join(trip_dir, trip_id + ".track")
    nodes_path = os.path.join(trip_dir, trip_id + ".nodes")
    arcs_path = os.path.join(trip_dir, trip_id + ".arcs")
    route_path = os.path.join(trip_dir, trip_id + ".route")

    if not all(os.path.exists(p) for p in [
        track_path, nodes_path, arcs_path, route_path
    ]):
        print("Skipping incomplete trip:", trip_id)
        continue

    track = np.loadtxt(track_path, ndmin=2)
    nodes = np.loadtxt(nodes_path, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)
    route = np.loadtxt(
        route_path, dtype=int, ndmin=2
    ).flatten()

    # Preserve original observation row order.
    gps_lon = track[:, 0]
    gps_lat = track[:, 1]
    timestamps = track[:, 2]

    # Select a local UTM projection for metre-based calculations.
    mean_lon = float(np.mean(gps_lon))
    mean_lat = float(np.mean(gps_lat))

    zone = int((mean_lon + 180) // 6) + 1
    epsg = (32600 if mean_lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        CRS.from_epsg(epsg),
        always_xy=True
    )

    # Project nodes and GPS coordinates into metres.
    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    gps_x, gps_y = transformer.transform(
        gps_lon, gps_lat
    )

    # Preserve arc direction and original arc IDs.
    valid_arc_mask = (
        (arcs[:, 0] >= 0) &
        (arcs[:, 0] < len(nodes)) &
        (arcs[:, 1] >= 0) &
        (arcs[:, 1] < len(nodes))
    )

    original_arc_ids = np.flatnonzero(valid_arc_mask)
    valid_arcs = arcs[valid_arc_mask]

    from_ids = valid_arcs[:, 0]
    to_ids = valid_arcs[:, 1]

    start_x = node_x[from_ids]
    start_y = node_y[from_ids]
    end_x = node_x[to_ids]
    end_y = node_y[to_ids]

    # Segment midpoints used for fast candidate lookup.
    midpoint_x = (start_x + end_x) / 2
    midpoint_y = (start_y + end_y) / 2

    midpoint_tree = cKDTree(
        np.column_stack((midpoint_x, midpoint_y))
    )

    search_k = min(
        MIDPOINT_SEARCH_K,
        len(valid_arcs)
    )

    route_arc_ids = set(route.tolist())

    # GPS movement heading from previous observation.
    gps_headings = np.full(len(track), np.nan)

    for i in range(1, len(track)):
        time_gap = timestamps[i] - timestamps[i - 1]

        if 0 < time_gap <= HEADING_GAP_LIMIT_SECONDS:
            gps_headings[i] = calculate_heading(
                gps_x[i - 1], gps_y[i - 1],
                gps_x[i], gps_y[i]
            )

    for observation_id in range(len(track)):
        point_x = gps_x[observation_id]
        point_y = gps_y[observation_id]

        # Find nearby segment midpoints.
        _, nearby_indices = midpoint_tree.query(
            [point_x, point_y],
            k=search_k
        )

        nearby_indices = np.atleast_1d(nearby_indices)

        candidate_distances = []

        for segment_index in nearby_indices:
            ax = start_x[segment_index]
            ay = start_y[segment_index]
            bx = end_x[segment_index]
            by = end_y[segment_index]

            dx = bx - ax
            dy = by - ay

            length_squared = dx * dx + dy * dy

            if length_squared == 0:
                fraction = 0.0
            else:
                fraction = (
                    (point_x - ax) * dx +
                    (point_y - ay) * dy
                ) / length_squared

                fraction = np.clip(fraction, 0.0, 1.0)

            closest_x = ax + fraction * dx
            closest_y = ay + fraction * dy

            distance = np.hypot(
                point_x - closest_x,
                point_y - closest_y
            )

            candidate_distances.append(
                (segment_index, distance)
            )

        # Select the nearest candidate segments.
        candidate_distances.sort(key=lambda item: item[1])

        for rank, (segment_index, distance) in enumerate(
            candidate_distances[:TOP_CANDIDATES],
            start=1
        ):
            arc_id = int(original_arc_ids[segment_index])

            road_heading = calculate_heading(
                start_x[segment_index],
                start_y[segment_index],
                end_x[segment_index],
                end_y[segment_index]
            )

            gps_heading = gps_headings[observation_id]

            if np.isfinite(gps_heading):
                heading_diff = angle_difference(
                    gps_heading,
                    road_heading
                )
            else:
                heading_diff = np.nan

            all_features.append({
                "trip_id": trip_id,
                "observation_id": observation_id,
                "gps_longitude": gps_lon[observation_id],
                "gps_latitude": gps_lat[observation_id],
                "gps_timestamp": timestamps[observation_id],
                "candidate_arc_id": arc_id,
                "from_node_id": int(from_ids[segment_index]),
                "to_node_id": int(to_ids[segment_index]),
                "distance_to_arc_m": float(distance),
                "gps_heading_deg": (
                    float(gps_heading)
                    if np.isfinite(gps_heading) else np.nan
                ),
                "road_heading_deg": float(road_heading),
                "heading_difference_deg": float(heading_diff)
                    if np.isfinite(heading_diff) else np.nan,
                "candidate_rank": rank,
                "is_in_reference_route": (
                    arc_id in route_arc_ids
                ),
                "is_ground_truth": pd.NA
            })

    print(
        "  Observations:", len(track),
        "| Candidates generated:",
        min(TOP_CANDIDATES, len(valid_arcs)) * len(track)
    )

# Save combined feature dataset.
features = pd.DataFrame(all_features)

features["is_ground_truth"] = features[
    "is_ground_truth"
].astype("boolean")

output_path = os.path.join(
    OUTPUT_DIR,
    "training_candidate_features.csv"
)

features.to_csv(output_path, index=False)

print("\nCANDIDATE FEATURE GENERATION COMPLETED")
print("Training trips processed:", features["trip_id"].nunique())
print("Total candidate rows:", len(features))
print("Unique GPS observations:", features[
    ["trip_id", "observation_id"]
].drop_duplicates().shape[0])
print("Output:", output_path)
print("\nSample rows:")
print(features.head(10).to_string(index=False))
