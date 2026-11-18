
import os
import numpy as np

SAMPLES = [
    ("train", "00000003"),
    ("test", "00000000")
]


def load_numeric_file(filepath):
    return np.loadtxt(filepath, dtype=float, ndmin=2)


def inspect_trip(partition, trip_id):

    folder = os.path.join("data", partition, trip_id)

    print("\n" + "=" * 65)
    print(f"PARTITION: {partition} | TRIP: {trip_id}")
    print("=" * 65)

    files = {
        "track": trip_id + ".track",
        "nodes": trip_id + ".nodes",
        "arcs": trip_id + ".arcs",
        "route": trip_id + ".route"
    }

    data = {}

    for name, filename in files.items():

        filepath = os.path.join(folder, filename)
        data[name] = load_numeric_file(filepath)

        print(f"\n{name.upper()}")
        print("Rows:", len(data[name]))
        print("Columns:", data[name].shape[1])
        print("Minimum:", np.min(data[name], axis=0))
        print("Maximum:", np.max(data[name], axis=0))
        print("Non-finite values:", np.sum(~np.isfinite(data[name])))

    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    print("\nADDITIONAL VALIDATION")
    print("Duplicate node coordinates:", 
          len(nodes) - len(np.unique(nodes, axis=0)))

    print("Duplicate arc rows:",
          len(arcs) - len(np.unique(arcs, axis=0)))

    print("Duplicate route values:",
          len(route) - len(np.unique(route)))

    print("Track observation indices increasing:",
          np.all(np.diff(track[:, 2]) > 0))

    print("\nFIRST 20 ROUTE VALUES:")
    print(route.flatten()[:20])

    print("\nFIRST 10 ARC ROWS:")
    print(arcs[:10])

    print("\nFIRST 10 NODE ROWS:")
    print(nodes[:10])


def main():

    for partition, trip_id in SAMPLES:
        inspect_trip(partition, trip_id)


if __name__ == "__main__":
    main()
