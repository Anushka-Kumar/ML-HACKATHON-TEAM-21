
import numpy as np
import os

SAMPLES = [
    ("train", "00000003"),
    ("test", "00000000")
]

for partition, trip_id in SAMPLES:
    folder = os.path.join("data", partition, trip_id)

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

    arc_ids = np.arange(len(arcs))

    print("\n" + "=" * 55)
    print("TRIP:", partition, trip_id)
    print("=" * 55)

    print("Total arc IDs:", len(arc_ids))
    print("Total route entries:", len(route))

    valid = np.isin(route, arc_ids)

    print("Valid route arc IDs:", valid.sum())
    print("Invalid route arc IDs:", (~valid).sum())

    print("Route IDs outside arc range:")
    print(route[~valid][:20])

    print("First 5 route arc connections:")

    for arc_id in route[:5]:
        if 0 <= arc_id < len(arcs):
            print(
                "Arc ID:", arc_id,
                "| From node:", arcs[arc_id, 0],
                "| To node:", arcs[arc_id, 1]
            )
