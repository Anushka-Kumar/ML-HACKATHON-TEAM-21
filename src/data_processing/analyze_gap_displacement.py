
import numpy as np

CASES = {
    "00000078": {
        "before": (3.595384, 44.321698),
        "after": (3.595358, 44.321711),
        "time_gap": 52810
    },
    "00000090": {
        "before": (10.445741, 51.621282),
        "after": (10.445779, 51.621300),
        "time_gap": 153064
    }
}


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


for trip_id, values in CASES.items():
    distance = haversine(
        *values["before"],
        *values["after"]
    )

    time_gap = values["time_gap"]
    speed = distance / time_gap

    print("\nTrip:", trip_id)
    print("Spatial displacement:", round(distance, 3), "metres")
    print("Time gap:", time_gap, "seconds")
    print("Implied average speed:", round(speed * 3.6, 6), "km/h")
