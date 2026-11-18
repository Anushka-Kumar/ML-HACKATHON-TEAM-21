
import os
import numpy as np

TRAIN_DIR = "data/train"

# Trips identified in the full validation report
TRIPS = [
    "00000078",
    "00000090",
    "00000069",
    "00000021",
    "00000011",
    "00000002",
    "00000019",
    "00000033",
    "00000041",
    "00000042",
    "00000075",
    "00000080",
    "00000086",
    "00000089"
]

for trip_id in TRIPS:
    folder = os.path.join(TRAIN_DIR, trip_id)

    track = np.loadtxt(
        os.path.join(folder, trip_id + ".track"),
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

    timestamps = track[:, 2]
    gaps = np.diff(timestamps)

    print("\n" + "=" * 60)
    print("TRIP:", trip_id)
    print("=" * 60)

    # Show unusually large GPS gaps
    large_gap_indices = np.where(gaps > 5)[0]

    print("\nGPS gaps greater than 5 seconds:")
    print("Number:", len(large_gap_indices))

    for index in large_gap_indices[:10]:
        print(
            "Between observations:", index, "and", index + 1,
            "| Gap:", gaps[index], "seconds",
            "| Time values:", timestamps[index],
            "to", timestamps[index + 1]
        )

    # Inspect disconnected route transitions
    route_arcs = arcs[route]

    disconnected = np.where(
        route_arcs[:-1, 1] != route_arcs[1:, 0]
    )[0]

    print("\nDisconnected route transitions:")
    print("Number:", len(disconnected))

    for index in disconnected[:10]:
        print(
            "Arc", route[index],
            "ends at node", route_arcs[index, 1],
            "| Next arc", route[index + 1],
            "starts at node", route_arcs[index + 1, 0]
        )
