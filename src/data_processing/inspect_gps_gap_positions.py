
import os
import numpy as np

TRIPS = {
    "00000078": 1,
    "00000090": 534
}

for trip_id, gap_index in TRIPS.items():

    filepath = os.path.join(
        "data", "train", trip_id, trip_id + ".track"
    )

    track = np.loadtxt(filepath, ndmin=2)

    print("\n" + "=" * 65)
    print("TRIP:", trip_id)
    print("=" * 65)

    start = max(0, gap_index - 3)
    end = min(len(track), gap_index + 5)

    for i in range(start, end):
        longitude, latitude, timestamp = track[i]

        print(
            f"Observation {i}: "
            f"Longitude={longitude:.6f}, "
            f"Latitude={latitude:.6f}, "
            f"Timestamp={timestamp:.3f}"
        )

    if gap_index + 1 < len(track):
        before = track[gap_index]
        after = track[gap_index + 1]

        print("\nGAP DETAILS")
        print("Time difference:", after[2] - before[2], "seconds")
        print(
            "Coordinate before:",
            before[0], before[1]
        )
        print(
            "Coordinate after:",
            after[0], after[1]
        )
