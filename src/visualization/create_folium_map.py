
import os
import json
import numpy as np
import folium

TRIP_ID = "00000003"
TRIP_DIR = os.path.join("data", "train", TRIP_ID)
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
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

# GPS coordinates
gps_lon = track[:, 0]
gps_lat = track[:, 1]

# Folium uses [latitude, longitude]
gps_locations = [
    [lat, lon]
    for lon, lat in zip(gps_lon, gps_lat)
]

# Create map centred on GPS trajectory
center = [
    float(np.mean(gps_lat)),
    float(np.mean(gps_lon))
]

m = folium.Map(
    location=center,
    zoom_start=14,
    tiles="OpenStreetMap"
)

# Road network as GeoJSON LineStrings
network_features = []

for arc_id, (from_node, to_node) in enumerate(arcs):
    start = nodes[from_node]
    end = nodes[to_node]

    network_features.append({
        "type": "Feature",
        "properties": {"arc_id": int(arc_id)},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [float(start[0]), float(start[1])],
                [float(end[0]), float(end[1])]
            ]
        }
    })

network_geojson = {
    "type": "FeatureCollection",
    "features": network_features
}

network_layer = folium.FeatureGroup(
    name="Road network",
    show=True
)

folium.GeoJson(
    network_geojson,
    style_function=lambda feature: {
        "color": "gray",
        "weight": 1,
        "opacity": 0.45
    },
    name="Road network"
).add_to(network_layer)

network_layer.add_to(m)

# Reference route
route_features = []

for arc_id in route:
    from_node, to_node = arcs[arc_id]

    start = nodes[from_node]
    end = nodes[to_node]

    route_features.append({
        "type": "Feature",
        "properties": {"arc_id": int(arc_id)},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [float(start[0]), float(start[1])],
                [float(end[0]), float(end[1])]
            ]
        }
    })

route_geojson = {
    "type": "FeatureCollection",
    "features": route_features
}

route_layer = folium.FeatureGroup(
    name="Reference route",
    show=True
)

folium.GeoJson(
    route_geojson,
    style_function=lambda feature: {
        "color": "blue",
        "weight": 5,
        "opacity": 0.9
    }
).add_to(route_layer)

route_layer.add_to(m)

# GPS trajectory
gps_layer = folium.FeatureGroup(
    name="GPS trajectory",
    show=True
)

folium.PolyLine(
    locations=gps_locations,
    color="red",
    weight=4,
    opacity=0.9
).add_to(gps_layer)

gps_layer.add_to(m)

# Start and end markers
folium.Marker(
    gps_locations[0],
    popup="GPS Start",
    tooltip="Start",
    icon=folium.Icon(color="green", icon="play")
).add_to(m)

folium.Marker(
    gps_locations[-1],
    popup="GPS End",
    tooltip="End",
    icon=folium.Icon(color="red", icon="stop")
).add_to(m)

# Layer controls
folium.LayerControl(collapsed=False).add_to(m)

# Fit map to GPS trajectory
m.fit_bounds(gps_locations)

# Save HTML
output_path = os.path.join(
    OUTPUT_DIR,
    f"trip_{TRIP_ID}_interactive_map.html"
)

m.save(output_path)

print("FOLIUM MAP CREATED SUCCESSFULLY")
print("Trip:", TRIP_ID)
print("GPS observations:", len(track))
print("Road-network arcs:", len(arcs))
print("Reference-route arcs:", len(route))
print("Saved to:", output_path)
