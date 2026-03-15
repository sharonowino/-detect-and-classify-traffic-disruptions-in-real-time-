"""
GTFS Real-Time Data Loader

Loads data from GTFS-RT feeds and transforms it to match the dashboard's expected format.
"""

import requests
from google.transit import gtfs_realtime_pb2
import pandas as pd
from datetime import datetime
from typing import Optional


# Default URLs (OVAPI - Netherlands)
DEFAULT_VEHICLE_URL = "http://gtfs.ovapi.nl/nl/vehiclePositions.pb"
DEFAULT_TRIP_UPDATE_URL = "http://gtfs.ovapi.nl/nl/tripUpdates.pb"
DEFAULT_ALERTS_URL = "http://gtfs.ovapi.nl/nl/alerts.pb"


def load_vehicle_positions(url: str = DEFAULT_VEHICLE_URL, timeout: int = 15) -> pd.DataFrame:
    """
    Load vehicle positions from GTFS-RT feed.
    
    Args:
        url: URL of the vehicle positions feed
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with columns: vehicle_id, lat, lon, timestamp, route_id, trip_id, speed
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        feed.ParseFromString(response.content)
    except Exception as e:
        print(f"Error fetching vehicle positions: {e}")
        return pd.DataFrame()
    
    vehicles = []
    
    for entity in feed.entity:
        if entity.HasField('vehicle'):
            v = entity.vehicle
            
            # Get position
            lat = None
            lon = None
            speed = None
            if v.HasField('position'):
                lat = v.position.latitude if v.position.HasField('latitude') else None
                lon = v.position.longitude if v.position.HasField('longitude') else None
                speed = v.position.speed if v.position.HasField('speed') else None
            
            # Get trip info
            route_id = None
            trip_id = None
            if v.HasField('trip'):
                route_id = v.trip.route_id if v.trip.HasField('route_id') else None
                trip_id = v.trip.trip_id if v.trip.HasField('trip_id') else None
            
            # Get vehicle ID
            vehicle_id = None
            if v.HasField('vehicle'):
                vehicle_id = v.vehicle.id if v.vehicle.HasField('id') else None
            
            # Get timestamp
            timestamp = None
            if v.HasField('timestamp'):
                timestamp = datetime.fromtimestamp(v.timestamp)
            
            vehicles.append({
                "vehicle_id": vehicle_id,
                "route_id": route_id,
                "trip_id": trip_id,
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "timestamp": timestamp
            })
    
    return pd.DataFrame(vehicles)


def load_trip_updates(url: str = DEFAULT_TRIP_UPDATE_URL, timeout: int = 15) -> pd.DataFrame:
    """
    Load trip updates (delay information) from GTFS-RT feed.
    
    Args:
        url: URL of the trip updates feed
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with columns: trip_id, route_id, stop_id, arrival_delay, departure_delay
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        feed.ParseFromString(response.content)
    except Exception as e:
        print(f"Error fetching trip updates: {e}")
        return pd.DataFrame()
    
    trips = []
    
    for entity in feed.entity:
        if entity.HasField('trip_update'):
            trip = entity.trip_update
            
            # Get trip info
            route_id = None
            trip_id = None
            if trip.HasField('trip'):
                route_id = trip.trip.route_id if trip.trip.HasField('route_id') else None
                trip_id = trip.trip.trip_id if trip.trip.HasField('trip_id') else None
            
            # Get stop time updates
            if trip.HasField('stop_time_update'):
                for stop in trip.stop_time_update:
                    arrival_delay = None
                    departure_delay = None
                    
                    if stop.HasField('arrival') and stop.arrival.HasField('delay'):
                        arrival_delay = stop.arrival.delay
                    if stop.HasField('departure') and stop.departure.HasField('delay'):
                        departure_delay = stop.departure.delay
                    
                    trips.append({
                        "trip_id": trip_id,
                        "route_id": route_id,
                        "stop_id": stop.stop_id if stop.HasField('stop_id') else None,
                        "arrival_delay": arrival_delay,
                        "departure_delay": departure_delay
                    })
    
    return pd.DataFrame(trips)


def load_alerts(url: str = DEFAULT_ALERTS_URL, timeout: int = 15) -> pd.DataFrame:
    """
    Load service alerts from GTFS-RT feed.
    
    Args:
        url: URL of the service alerts feed
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with columns: cause, effect, description, route_id
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        feed.ParseFromString(response.content)
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        return pd.DataFrame()
    
    alerts = []
    
    for entity in feed.entity:
        if entity.HasField('alert'):
            alert = entity.alert
            
            # Get cause and effect
            cause = alert.cause if alert.HasField('cause') else None
            effect = alert.effect if alert.HasField('effect') else None
            
            # Get description text
            description = None
            if alert.HasField('header_text'):
                for translation in alert.header_text.translation:
                    description = translation.text
                    break
            
            # Get affected routes
            route_id = None
            if alert.HasField('informed_entity'):
                for entity_info in alert.informed_entity:
                    if entity_info.HasField('route_id'):
                        route_id = entity_info.route_id
                        break
            
            alerts.append({
                "route_id": route_id,
                "cause": cause,
                "effect": effect,
                "description": description
            })
    
    return pd.DataFrame(alerts)


def load_all_gtfs_data(
    vehicle_url: str = DEFAULT_VEHICLE_URL,
    trip_url: str = DEFAULT_TRIP_UPDATE_URL,
    alerts_url: str = DEFAULT_ALERTS_URL
) -> tuple:
    """
    Load all three GTFS-RT feeds.
    
    Returns:
        Tuple of (vehicle_df, trip_updates_df, alerts_df)
    """
    vehicle_df = load_vehicle_positions(vehicle_url)
    trip_df = load_trip_updates(trip_url)
    alerts_df = load_alerts(alerts_url)
    
    return vehicle_df, trip_df, alerts_df


def transform_to_dashboard_format(
    vehicle_df: pd.DataFrame,
    trip_df: pd.DataFrame,
    alerts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform GTFS-RT data to match the dashboard's expected schema.
    
    Expected output columns:
    - route_id, timestamp, speed_mean, speed_std
    - delay_mean_5m, delay_mean_15m, delay_mean_30m
    - bunching_index, on_time_pct, headway_variance
    - alert_nlp_score, alert_count, fleet_utilization
    - speed_drop_ratio, lat, lon
    """
    if vehicle_df.empty:
        return pd.DataFrame()
    
    # Aggregate vehicle positions by route
    agg_dict = {}
    if 'vehicle_id' in vehicle_df.columns:
        agg_dict['vehicle_id'] = 'count'
    if 'lat' in vehicle_df.columns:
        agg_dict['lat'] = 'mean'
    if 'lon' in vehicle_df.columns:
        agg_dict['lon'] = 'mean'
    if 'speed' in vehicle_df.columns:
        agg_dict['speed'] = ['mean', 'std']
    
    if not agg_dict:
        return pd.DataFrame()
    
    try:
        df = vehicle_df.groupby('route_id').agg(agg_dict).reset_index()
    except Exception as e:
        print(f"Aggregation error: {e}")
        return pd.DataFrame()
    
    # Flatten column names if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    
    # Rename columns to match dashboard schema
    rename_map = {}
    for col in df.columns:
        if 'vehicle_id' in col:
            rename_map[col] = 'vehicle_count'
        elif 'lat' in col and 'mean' in col:
            rename_map[col] = 'lat'
        elif 'lon' in col and 'mean' in col:
            rename_map[col] = 'lon'
        elif 'speed' in col and 'mean' in col:
            rename_map[col] = 'speed_mean'
        elif 'speed' in col and 'std' in col:
            rename_map[col] = 'speed_std'
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    if 'vehicle_count' not in df.columns:
        df['vehicle_count'] = 1
    if 'lat' not in df.columns:
        df['lat'] = 52.0
    if 'lon' not in df.columns:
        df['lon'] = 4.5
    if 'speed_mean' not in df.columns:
        df['speed_mean'] = 30.0
    if 'speed_std' not in df.columns:
        df['speed_std'] = 5.0
    
    # Add timestamp
    df['timestamp'] = datetime.now()
    
    # Merge with trip updates to get delay information
    if not trip_df.empty and 'route_id' in trip_df.columns:
        delay_agg = trip_df.groupby('route_id').agg({
            'arrival_delay': 'mean',
            'departure_delay': 'mean'
        }).reset_index()
        
        delay_agg = delay_agg.rename(columns={
            'arrival_delay': 'delay_mean_5m',
            'departure_delay': 'delay_mean_15m'
        })
        
        df = df.merge(delay_agg, on='route_id', how='left')
        
        # Fill missing delays with reasonable defaults
        df['delay_mean_5m'] = df['delay_mean_5m'].fillna(0)
        df['delay_mean_15m'] = df['delay_mean_15m'].fillna(df['delay_mean_5m'] + 10)
        df['delay_mean_30m'] = df['delay_mean_15m'] + 10
    else:
        df['delay_mean_5m'] = 0
        df['delay_mean_15m'] = 10
        df['delay_mean_30m'] = 20
    
    # Merge with alerts to get alert counts
    if not alerts_df.empty and 'route_id' in alerts_df.columns:
        alert_counts = alerts_df.groupby('route_id').size().reset_index(name='alert_count')
        df = df.merge(alert_counts, on='route_id', how='left')
        df['alert_count'] = df['alert_count'].fillna(0)
        df['alert_nlp_score'] = (df['alert_count'] / df['alert_count'].max()).fillna(0) if df['alert_count'].max() > 0 else 0
    else:
        df['alert_count'] = 0
        df['alert_nlp_score'] = 0
    
    # Compute derived features
    df['bunching_index'] = (df['speed_std'] / (df['speed_mean'] + 1)).clip(0, 1)
    df['on_time_pct'] = (1 - (df['delay_mean_15m'] / 300)).clip(0, 1).fillna(0.85)
    df['headway_variance'] = (60 / df['vehicle_count'].clip(1)).fillna(30)
    
    expected_fleet = 10
    df['fleet_utilization'] = (df['vehicle_count'] / expected_fleet).clip(0.5, 1.0)
    
    df['speed_drop_ratio'] = ((50 - df['speed_mean']).clip(0, 30) / 30).fillna(0.1)
    
    # Select final columns in correct order
    final_cols = [
        'route_id', 'timestamp', 'speed_mean', 'speed_std',
        'delay_mean_5m', 'delay_mean_15m', 'delay_mean_30m',
        'bunching_index', 'on_time_pct', 'headway_variance',
        'alert_nlp_score', 'alert_count', 'fleet_utilization',
        'speed_drop_ratio', 'lat', 'lon'
    ]
    
    # Only select columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]
    
    return df


if __name__ == "__main__":
    # Test the loader
    print("Loading GTFS data...")
    vehicle_df, trip_df, alerts_df = load_all_gtfs_data()
    print(f"Vehicles: {len(vehicle_df)}")
    print(f"Trip updates: {len(trip_df)}")
    print(f"Alerts: {len(alerts_df)}")
    
    print("\\nTransforming to dashboard format...")
    df = transform_to_dashboard_format(vehicle_df, trip_df, alerts_df)
    print(f"Routes: {len(df)}")
    print(df.head())
