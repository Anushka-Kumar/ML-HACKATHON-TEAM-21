
import os
import csv
import numpy as np

TRAIN_DIR = "data/train"
OUTPUT_DIR = "results/eda"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "training_trip_validation.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_file(path, columns):
    data = np.loadtxt(path, dtype=float, ndmin=2)

    if data.shape[1] != columns:
        raise ValueError(
            f"Expected {columns} columns, found {data.shape[1]}"
        )

    return data


def count_weak_components(node_count, arcs):
    """Count connected components treating arcs as undirected."""
    parent = list(range(node_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for from_node, to_node in arcs.astype(int):
        union(from_node, to_node)

    roots = {find(i) for i in range(node_count)}
    return len(roots)


def validate_trip(trip_id):
    folder = os.path.join(TRAIN_DIR, trip_id)

    track = load_file(
        os.path.join(folder, trip_id + ".track"), 3
    )
    nodes = load_file(
        os.path.join(folder, trip_id + ".nodes"), 2
    )
    arcs = load_file(
        os.path.join(folder, trip_id + ".arcs"), 2
    )
    route = load_file(
        os.path.join(folder, trip_id + ".route"), 1
    ).flatten()

    # File-level validity
    finite_track = np.all(np.isfinite(track))
    finite_nodes = np.all(np.isfinite(nodes))
    finite_arcs = np.all(np.isfinite(arcs))
    finite_route = np.all(np.isfinite(route))

    # Official coordinate order: longitude, latitude
    longitude = track[:, 0]
    latitude = track[:, 1]
    timestamps = track[:, 2]

    valid_coordinates = (
        np.all((longitude >= -180) & (longitude <= 180))
        and np.all((latitude >= -90) & (latitude <= 90))
    )

    # Timestamp checks
    time_differences = np.diff(timestamps)
    timestamps_increasing = np.all(time_differences > 0)

    irregular_gaps = np.sum(
        ~np.isclose(time_differences, 1.0)
    ) if len(time_differences) else 0

    largest_time_gap = (
        float(np.max(time_differences))
        if len(time_differences) else 0
    )

    # Node and arc checks
    node_count = len(nodes)
    arc_count = len(arcs)

    integer_arcs = np.all(arcs == np.floor(arcs))
    arc_endpoints_valid = (
        integer_arcs
        and np.all(arcs >= 0)
        and np.all(arcs < node_count)
    )

    # Route values are arc row indices (arc IDs)
    integer_route = np.all(route == np.floor(route))
    route_ids_valid = (
        integer_route
        and np.all(route >= 0)
        and np.all(route < arc_count)
    )

    # Consecutive route arcs should connect
    connected_transitions = 0
    disconnected_transitions = 0

    if route_ids_valid and len(route) > 1:
        route_arcs = arcs[route.astype(int)]

        transitions = (
            route_arcs[:-1, 1] == route_arcs[1:, 0]
        )

        connected_transitions = int(np.sum(transitions))
        disconnected_transitions = int(
            len(transitions) - np.sum(transitions)
        )

    # Road network connectivity (weak connectivity)
    if arc_endpoints_valid:
        integer_arc_data = arcs.astype(int)

        degree = np.zeros(node_count, dtype=int)

        for start, end in integer_arc_data:
            degree[start] += 1
            degree[end] += 1

        isolated_nodes = int(np.sum(degree == 0))
        components = count_weak_components(
            node_count, integer_arc_data
        )
    else:
        isolated_nodes = -1
        components = -1

    return {
        "trip_id": trip_id,
        "track_rows": len(track),
        "node_count": node_count,
        "arc_count": arc_count,
        "route_entries": len(route),
        "finite_values": (
            finite_track and finite_nodes
            and finite_arcs and finite_route
        ),
        "valid_gps_coordinates": valid_coordinates,
        "timestamps_increasing": timestamps_increasing,
        "irregular_time_gaps": int(irregular_gaps),
        "largest_time_gap_seconds": largest_time_gap,
        "arc_endpoints_valid": arc_endpoints_valid,
        "route_arc_ids_valid": route_ids_valid,
        "connected_route_transitions": connected_transitions,
        "disconnected_route_transitions": disconnected_transitions,
        "isolated_nodes": isolated_nodes,
        "weakly_connected_components": components
    }


def main():
    trips = sorted(
        folder for folder in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, folder))
    )

    results = []

    print("PATHSNAP - TRAINING DATA VALIDATION")
    print("Training trips found:", len(trips))

    for index, trip_id in enumerate(trips, start=1):
        try:
            result = validate_trip(trip_id)
            results.append(result)

            print(
                f"[{index}/{len(trips)}] {trip_id} "
                f"| GPS: {result['track_rows']} "
                f"| Nodes: {result['node_count']} "
                f"| Arcs: {result['arc_count']} "
                f"| Route: {result['route_entries']} "
                f"| Route IDs valid: {result['route_arc_ids_valid']}"
            )

        except Exception as error:
            print(
                f"[{index}/{len(trips)}] ERROR in {trip_id}: {error}"
            )

    if not results:
        print("No trips could be validated.")
        return

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== SUMMARY ==========")
    print("Successfully inspected:", len(results))
    print("Total training folders:", len(trips))

    print(
        "Trips with valid route IDs:",
        sum(r["route_arc_ids_valid"] for r in results)
    )
    print(
        "Trips with valid arc endpoints:",
        sum(r["arc_endpoints_valid"] for r in results)
    )
    print(
        "Trips with increasing timestamps:",
        sum(r["timestamps_increasing"] for r in results)
    )
    print(
        "Trips with finite values:",
        sum(r["finite_values"] for r in results)
    )

    print("\nCSV saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
