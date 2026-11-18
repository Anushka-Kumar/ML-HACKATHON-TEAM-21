
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TRIP_ID = "00000003"
TRIP_DIR = os.path.join("data", "train", TRIP_ID)
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the four dataset files
track = np.loadtxt(
    os.path.join(TRIP_DIR, TRIP_ID + ".track"),
    ndmin=2
)

nodes = np.loadtxt(
    os.path.join(TRIP_DIR, TRIP_ID + ".nodes"),
    ndmin=2
)

arcs = np.loadtxt(
    os.path.join(TRIP_DIR, TRIP_ID + ".arcs"),
    dtype=int,
    ndmin=2
)

route = np.loadtxt(
    os.path.join(TRIP_DIR, TRIP_ID + ".route"),
    dtype=int,
    ndmin=2
).flatten()

# Official coordinate order: longitude, latitude
gps_lon = track[:, 0]
gps_lat = track[:, 1]

node_lon = nodes[:, 0]
node_lat = nodes[:, 1]

# Create all road-network segments
network_segments = np.stack(
    [
        nodes[arcs[:, 0]],
        nodes[arcs[:, 1]]
    ],
    axis=1
)

# Create reference-route segments in chronological order
route_arcs = arcs[route]

route_segments = np.stack(
    [
        nodes[route_arcs[:, 0]],
        nodes[route_arcs[:, 1]]
    ],
    axis=1
)

# Create figure
fig, ax = plt.subplots(figsize=(12, 9))

# Plot road network
network = LineCollection(
    network_segments,
    colors="lightgray",
    linewidths=0.35,
    alpha=0.6,
    label="Road network"
)
ax.add_collection(network)

# Plot reference route
reference = LineCollection(
    route_segments,
    colors="blue",
    linewidths=2.2,
    label="Reference route"
)
ax.add_collection(reference)

# Plot GPS trajectory
ax.plot(
    gps_lon,
    gps_lat,
    color="red",
    linewidth=1.2,
    marker=".",
    markersize=3,
    label="GPS trajectory",
    zorder=4
)

# Mark start and end GPS positions
ax.scatter(
    gps_lon[0], gps_lat[0],
    color="green",
    s=70,
    label="GPS start",
    zorder=5
)

ax.scatter(
    gps_lon[-1], gps_lat[-1],
    color="black",
    marker="x",
    s=70,
    label="GPS end",
    zorder=5
)

ax.autoscale()
ax.set_aspect("equal", adjustable="datalim")

ax.set_title(
    f"PathSnap - GPS Trajectory and Road Network\nTrip {TRIP_ID}"
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    f"trip_{TRIP_ID}_trajectory.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("VISUALIZATION COMPLETED")
print("Trip:", TRIP_ID)
print("GPS observations:", len(track))
print("Road-network nodes:", len(nodes))
print("Road-network arcs:", len(arcs))
print("Reference-route arcs:", len(route))
print("Saved to:", output_path)

plt.show()
