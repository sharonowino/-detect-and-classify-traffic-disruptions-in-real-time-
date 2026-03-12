import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2
from datetime import datetime

VEHICLE_URL = "YOUR_GTFS_VEHICLE_URL"

def fetch_vehicle_positions():

    feed = gtfs_realtime_pb2.FeedMessage()

    response = requests.get(VEHICLE_URL)
    feed.ParseFromString(response.content)

    rows = []

    for entity in feed.entity:

        if entity.HasField("vehicle"):

            vehicle = entity.vehicle

            rows.append({
                "vehicle_id": vehicle.vehicle.id,
                "lat": vehicle.position.latitude,
                "lon": vehicle.position.longitude,
                "timestamp": vehicle.timestamp,
                "retrieved_at": datetime.utcnow()
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    df = fetch_vehicle_positions()

    if len(df) > 0:
        df.to_csv("data/vehicle_positions.csv", mode="a", header=False, index=False)
