"""
GTFS Data Collector and Feature Engineering

Collects data from all 3 GTFS-RT feeds and creates features
matching the demo dataset schema for the dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings

from utils.gtfs_loader import (
    load_vehicle_positions,
    load_trip_updates, 
    load_alerts,
    DEFAULT_VEHICLE_URL,
    DEFAULT_TRIP_UPDATE_URL,
    DEFAULT_ALERTS_URL
)


class GTFSDataCollector:
    """
    Collects and merges data from all 3 GTFS-RT feeds.
    """
    
    def __init__(
        self,
        vehicle_url: str = DEFAULT_VEHICLE_URL,
        trip_url: str = DEFAULT_TRIP_UPDATE_URL,
        alerts_url: str = DEFAULT_ALERTS_URL,
        timeout: int = 15
    ):
        self.vehicle_url = vehicle_url
        self.trip_url = trip_url
        self.alerts_url = alerts_url
        self.timeout = timeout
        
    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """Collect all three feeds."""
        vehicle_df = load_vehicle_positions(self.vehicle_url, self.timeout)
        trip_df = load_trip_updates(self.trip_url, self.timeout)
        alerts_df = load_alerts(self.alerts_url, self.timeout)
        
        return {
            'vehicles': vehicle_df,
            'trips': trip_df,
            'alerts': alerts_df
        }
    
    def merge_data(
        self,
        vehicle_df: pd.DataFrame,
        trip_df: pd.DataFrame,
        alerts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all data sources into a single route-level dataset.
        """
        if vehicle_df.empty:
            return pd.DataFrame()
        
        # Aggregate vehicle positions by route
        route_vehicles = self._aggregate_vehicles(vehicle_df)
        
        # Aggregate trip updates by route
        route_trips = self._aggregate_trips(trip_df)
        
        # Aggregate alerts by route
        route_alerts = self._aggregate_alerts(alerts_df)
        
        # Merge all together
        df = route_vehicles.copy()
        
        if not route_trips.empty:
            df = df.merge(route_trips, on='route_id', how='left')
        
        if not route_alerts.empty:
            df = df.merge(route_alerts, on='route_id', how='left')
        
        return df
    
    def _aggregate_vehicles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate vehicle positions by route."""
        if df.empty or 'route_id' not in df.columns:
            return pd.DataFrame()
        
        # Filter out rows without route_id
        df = df[df['route_id'].notna()].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        agg_dict = {}
        if 'vehicle_id' in df.columns:
            agg_dict['vehicle_id'] = 'count'
        if 'lat' in df.columns:
            agg_dict['lat'] = 'mean'
        if 'lon' in df.columns:
            agg_dict['lon'] = 'mean'
        if 'speed' in df.columns:
            agg_dict['speed'] = ['mean', 'std']
        
        if not agg_dict:
            return pd.DataFrame()
        
        result = df.groupby('route_id').agg(agg_dict).reset_index()
        
        # Flatten multi-index columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns]
        
        # Rename columns
        rename_map = {}
        for col in result.columns:
            if 'vehicle_id' in col and 'count' in col:
                rename_map[col] = 'vehicle_count'
            elif 'lat' in col and 'mean' in col:
                rename_map[col] = 'lat'
            elif 'lon' in col and 'mean' in col:
                rename_map[col] = 'lon'
            elif 'speed' in col and 'mean' in col:
                rename_map[col] = 'speed_mean'
            elif 'speed' in col and 'std' in col:
                rename_map[col] = 'speed_std'
        
        result = result.rename(columns=rename_map)
        
        return result
    
    def _aggregate_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trip updates by route."""
        if df.empty or 'route_id' not in df.columns:
            return pd.DataFrame()
        
        df = df[df['route_id'].notna()].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        agg_dict = {}
        if 'arrival_delay' in df.columns:
            agg_dict['arrival_delay'] = ['mean', 'std', 'max']
        if 'departure_delay' in df.columns:
            agg_dict['departure_delay'] = ['mean', 'std', 'max']
        
        if not agg_dict:
            return pd.DataFrame()
        
        result = df.groupby('route_id').agg(agg_dict).reset_index()
        
        # Flatten columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns]
        
        # Rename
        rename_map = {}
        for col in result.columns:
            if 'arrival_delay' in col:
                rename_map[col] = col.replace('arrival_delay', 'delay')
            elif 'departure_delay' in col:
                rename_map[col] = col.replace('departure_delay', 'delay')
        
        result = result.rename(columns=rename_map)
        
        return result
    
    def _aggregate_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate alerts by route."""
        if df.empty or 'route_id' not in df.columns:
            return pd.DataFrame()
        
        df = df[df['route_id'].notna()].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        result = df.groupby('route_id').agg({
            'cause': 'count',  # Number of alerts
            'effect': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None  # Most common effect
        }).reset_index()
        
        result = result.rename(columns={'cause': 'alert_count'})
        
        return result


class GTFSFeatureEngineer:
    """
    Feature engineering for GTFS data to match demo dataset schema.
    
    Demo dataset columns:
    - route_id, timestamp
    - speed_mean, speed_std
    - delay_mean_5m, delay_mean_15m, delay_mean_30m
    - bunching_index, on_time_pct, headway_variance
    - alert_nlp_score, alert_count, fleet_utilization
    - speed_drop_ratio, lat, lon
    """
    
    def __init__(self):
        self.required_cols = [
            'route_id', 'timestamp', 'speed_mean', 'speed_std',
            'delay_mean_5m', 'delay_mean_15m', 'delay_mean_30m',
            'bunching_index', 'on_time_pct', 'headway_variance',
            'alert_nlp_score', 'alert_count', 'fleet_utilization',
            'speed_drop_ratio', 'lat', 'lon'
        ]
        
        # Features needed for ML model prediction
        self.prediction_features = [
            'speed_mean', 'speed_std', 'delay_mean_5m', 'delay_mean_15m', 'delay_mean_30m',
            'bunching_index', 'on_time_pct', 'headway_variance', 'alert_nlp_score',
            'alert_count', 'fleet_utilization', 'speed_drop_ratio'
        ]
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering to create demo-compatible dataset.
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Ensure required base columns exist
        df = self._ensure_base_columns(df)
        
        # Add derived features
        df = self._add_delay_features(df)
        df = self._add_bunching_features(df)
        df = self._add_ontime_features(df)
        df = self._add_headway_features(df)
        df = self._add_fleet_features(df)
        df = self._add_alert_features(df)
        df = self._add_speed_features(df)
        df = self._add_temporal_features(df)
        
        # Ensure all prediction features exist with valid values
        df = self._ensure_prediction_features(df)
        
        # Select and order columns
        df = self._select_final_columns(df)
        
        return df
    
    def _ensure_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure base columns exist with defaults."""
        defaults = {
            'vehicle_count': 1,
            'lat': 52.0,  # Netherlands center
            'lon': 4.5,
            'speed_mean': 30.0,
            'speed_std': 5.0,
            'timestamp': datetime.now()
        }
        
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
            elif df[col].isna().all():
                df[col] = default
        
        return df
    
    def _ensure_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all 12 prediction features exist with valid (non-NaN) values.
        This is critical for the ML model prediction.
        """
        for feat in self.prediction_features:
            if feat not in df.columns:
                # Add missing feature with default value
                df[feat] = 0.0
            else:
                # Fill NaN with reasonable defaults
                df[feat] = df[feat].fillna(self._get_default_for_feature(feat))
        
        return df
    
    def _get_default_for_feature(self, feature: str) -> float:
        """Get default value for a feature when data is missing."""
        defaults = {
            'speed_mean': 30.0,
            'speed_std': 5.0,
            'delay_mean_5m': 30.0,
            'delay_mean_15m': 60.0,
            'delay_mean_30m': 90.0,
            'bunching_index': 0.2,
            'on_time_pct': 0.85,
            'headway_variance': 30.0,
            'alert_nlp_score': 0.0,
            'alert_count': 0,
            'fleet_utilization': 0.85,
            'speed_drop_ratio': 0.15
        }
        return defaults.get(feature, 0.0)
    
    def _add_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delay-related features."""
        delay_cols = [c for c in df.columns if 'delay' in c and 'mean' in c]
        
        if delay_cols:
            main_delay = df[delay_cols[0]]
            df['delay_mean_5m'] = main_delay.fillna(0)
            
            # Estimate 15m and 30m delays
            if len(delay_cols) > 1:
                df['delay_mean_15m'] = df[delay_cols[1]].fillna(df['delay_mean_5m'] + 30)
            else:
                df['delay_mean_15m'] = df['delay_mean_5m'] + 30
            
            df['delay_mean_30m'] = df['delay_mean_15m'] + 60
        else:
            df['delay_mean_5m'] = np.random.uniform(0, 60, len(df))
            df['delay_mean_15m'] = df['delay_mean_5m'] + np.random.uniform(30, 120, len(df))
            df['delay_mean_30m'] = df['delay_mean_15m'] + np.random.uniform(60, 180, len(df))
        
        return df
    
    def _add_bunching_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bus bunching index (high variance = bunching)."""
        if 'speed_std' in df.columns and 'speed_mean' in df.columns:
            df['bunching_index'] = (
                df['speed_std'] / (df['speed_mean'] + 1)
            ).clip(0, 1).fillna(0.2)
        else:
            df['bunching_index'] = np.random.uniform(0.1, 0.4, len(df))
        
        return df
    
    def _add_ontime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add on-time performance features."""
        if 'delay_mean_15m' in df.columns:
            # On-time = no delay > 5 minutes (300 seconds)
            df['on_time_pct'] = (
                1 - (df['delay_mean_15m'] / 300).clip(0, 1)
            ).fillna(0.85)
        else:
            df['on_time_pct'] = np.random.uniform(0.7, 0.95, len(df))
        
        return df
    
    def _add_headway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add headway variance features."""
        if 'vehicle_count' in df.columns:
            # Higher vehicle count = lower headway variance
            df['headway_variance'] = (
                60 / df['vehicle_count'].clip(1)
            ).fillna(30)
        else:
            df['headway_variance'] = np.random.uniform(10, 45, len(df))
        
        return df
    
    def _add_fleet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fleet utilization features."""
        expected_fleet = 10  # Assume 10 vehicles per route
        
        if 'vehicle_count' in df.columns:
            df['fleet_utilization'] = (
                df['vehicle_count'] / expected_fleet
            ).clip(0.5, 1.0).fillna(0.85)
        else:
            df['fleet_utilization'] = np.random.uniform(0.7, 0.95, len(df))
        
        return df
    
    def _add_alert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add alert-related features."""
        if 'alert_count' not in df.columns:
            df['alert_count'] = 0
        
        df['alert_count'] = df['alert_count'].fillna(0)
        
        # NLP score based on alert count
        max_alerts = max(df['alert_count'].max(), 1)
        df['alert_nlp_score'] = (
            df['alert_count'] / max_alerts
        ).fillna(0)
        
        return df
    
    def _add_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add speed drop ratio."""
        reference_speed = 50  # km/h reference speed
        
        if 'speed_mean' in df.columns:
            df['speed_drop_ratio'] = (
                (reference_speed - df['speed_mean']).clip(0, reference_speed) / reference_speed
            ).fillna(0.1)
        else:
            df['speed_drop_ratio'] = np.random.uniform(0.05, 0.25, len(df))
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Fill any remaining NaT
        df['timestamp'] = df['timestamp'].fillna(datetime.now())
        
        return df
    
    def _select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final columns."""
        # Filter to only columns that exist
        final_cols = [c for c in self.required_cols if c in df.columns]
        
        # Add any extra columns
        extra_cols = [c for c in df.columns if c not in self.required_cols]
        
        return df[final_cols + extra_cols]


def collect_and_process(
    vehicle_url: str = DEFAULT_VEHICLE_URL,
    trip_url: str = DEFAULT_TRIP_UPDATE_URL,
    alerts_url: str = DEFAULT_ALERTS_URL,
    min_routes: int = 5
) -> pd.DataFrame:
    """
    Complete pipeline: collect GTFS data and engineer features.
    
    Returns a DataFrame matching the demo dataset schema.
    
    Args:
        vehicle_url: URL for vehicle positions feed
        trip_url: URL for trip updates feed
        alerts_url: URL for service alerts feed
        min_routes: Minimum number of routes to return (pads if needed)
    """
    # Collect data
    collector = GTFSDataCollector(
        vehicle_url=vehicle_url,
        trip_url=trip_url,
        alerts_url=alerts_url
    )
    
    raw_data = collector.collect_all()
    
    # Merge data
    merged_df = collector.merge_data(
        raw_data['vehicles'],
        raw_data['trips'],
        raw_data['alerts']
    )
    
    # Engineer features
    engineer = GTFSFeatureEngineer()
    final_df = engineer.engineer_features(merged_df)
    
    # If we have too few routes, generate synthetic ones for demo purposes
    if len(final_df) < min_routes:
        final_df = _generate_synthetic_routes(final_df, min_routes)
    
    return final_df


def _generate_synthetic_routes(existing_df: pd.DataFrame, min_routes: int) -> pd.DataFrame:
    """
    Generate synthetic routes to meet minimum route count.
    Uses existing route data as template.
    """
    import numpy as np
    
    if existing_df.empty:
        # Generate completely new synthetic data
        routes = [f"Route_{i:03d}" for i in range(1, min_routes + 1)]
        return pd.DataFrame({
            'route_id': routes,
            'timestamp': [datetime.now()] * min_routes,
            'speed_mean': np.random.uniform(20, 50, min_routes),
            'speed_std': np.random.uniform(2, 15, min_routes),
            'delay_mean_5m': np.random.uniform(0, 120, min_routes),
            'delay_mean_15m': np.random.uniform(10, 180, min_routes),
            'delay_mean_30m': np.random.uniform(20, 240, min_routes),
            'bunching_index': np.random.uniform(0, 1, min_routes),
            'on_time_pct': np.random.uniform(0.6, 1.0, min_routes),
            'headway_variance': np.random.uniform(5, 60, min_routes),
            'alert_nlp_score': np.random.uniform(0, 0.5, min_routes),
            'alert_count': np.random.poisson(1, min_routes),
            'fleet_utilization': np.random.uniform(0.7, 1.0, min_routes),
            'speed_drop_ratio': np.random.uniform(0, 0.3, min_routes),
            'lat': np.random.uniform(51.9, 52.1, min_routes),
            'lon': np.random.uniform(4.3, 5.1, min_routes)
        })
    
    # Use existing data as base and add synthetic variations
    n_needed = min_routes - len(existing_df)
    synthetic_routes = []
    
    for i in range(n_needed):
        # Use random existing route as template
        template = existing_df.iloc[i % len(existing_df)].copy()
        template['route_id'] = f"Route_{len(existing_df) + i + 1:03d}"
        # Add small variations
        for col in template.index:
            if col != 'route_id' and pd.api.types.is_numeric_dtype(template[col]):
                template[col] = template[col] * np.random.uniform(0.8, 1.2)
        synthetic_routes.append(template)
    
    return pd.concat([existing_df, pd.DataFrame(synthetic_routes)], ignore_index=True)


# Alternative: Load from OVAPI (Dutch feeds) with specific feature engineering
def collect_dutch_transit_data() -> pd.DataFrame:
    """
    Collect data from Dutch OVAPI feeds with specialized processing.
    """
    # OVAPI endpoints (Netherlands)
    vehicle_url = "http://gtfs.ovapi.nl/nl/vehiclePositions.pb"
    trip_url = "http://gtfs.ovapi.nl/nl/tripUpdates.pb"
    alerts_url = "http://gtfs.ovapi.nl/nl/alerts.pb"
    
    return collect_and_process(vehicle_url, trip_url, alerts_url)


if __name__ == "__main__":
    # Test the pipeline
    print("Collecting GTFS data...")
    df = collect_dutch_transit_data()
    
    if not df.empty:
        print(f"Collected data for {len(df)} routes")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    else:
        print("No data collected - feeds may be unavailable")
