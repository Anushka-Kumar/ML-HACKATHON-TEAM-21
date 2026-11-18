
import os
import numpy as np

TRAIN_DIR = "data/train"

TRIPS = [
    "00000002",
    "00000019",
    "00000033",
    "00000041",
    "00000042",
    "00000069",
    "00000075",
    "00000080",
    "00000086",
    "00000089"
]


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


for trip_id in TRIPS:
    folder = os.path.join(TRAIN_DIR, trip_id)

    nodes = np.loadtxt(
        os.path.join(folder, trip_id + ".nodes"),
        ndmin=2
    )

    arcs = np.loadtxt(
        os.path.join(folder, trip_id + ".arcs"),
        dtype=int,
        ndmin=2
    )

    route = np.loadtxt(
        os.path.join(folder, trip_id + ".route"),
        dtype=int,
        ndmin=2
    ).flatten()

    route_arcs = arcs[route]

    disconnected = np.where(
        route_arcs[:-1, 1] != route_arcs[1:, 0]
    )[0]

    if len(disconnected) == 0:
        continue

    print("\n" + "=" * 60)
    print("TRIP:", trip_id)
    print("=" * 60)

    for index in disconnected:
        end_node = route_arcs[index, 1]
        start_node = route_arcs[index + 1, 0]

        end_lon, end_lat = nodes[end_node]
        start_lon, start_lat = nodes[start_node]

        distance = haversine(
            end_lon, end_lat,
            start_lon, start_lat
        )

        print(
            f"Route position: {index} -> {index + 1}"
        )
        print(
            f"Arc IDs: {route[index]} -> {route[index + 1]}"
        )
        print(
            f"Node IDs: {end_node} -> {start_node}"
        )
        print(
            f"Geographical distance: {distance:.2f} metres"
        )
        print("-" * 40)


# Inspect the largest GPS gaps in the two most anomalous trips
for trip_id in ["00000078", "00000090"]:
    track = np.loadtxt(
        os.path.join(TRAIN_DIR, trip_id, trip_id + ".track"),
        ndmin=2
    )

    gaps = np.diff(track[:, 2])
    largest = np.argsort(gaps)[-10:][::-1]

    print("\nLARGEST GPS GAPS:", trip_id)

    for index in largest:
        print(
            f"Observations {index} -> {index + 1}: "
            f"{gaps[index]:.3f} seconds"
        )
