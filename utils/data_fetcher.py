"""Utility module to fetch data from GitHub"""
import pandas as pd
import os
import requests
from io import BytesIO

# GitHub raw file URLs
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sharonowino/-detect-and-classify-traffic-disruptions-in-real-time-/main/data"

DATA_FILES = {
    "merged": "merged_with_alerts.parquet",
    "vehicle": "vehicle_df.parquet",
    "alerts": "processed_df.parquet",
    "trip_updates": "df_trip_updates.parquet"
}

# Cache for downloaded data
_data_cache = {}

def clean_dataframe(df):
    """Clean dataframe by converting problematic types to JSON-serializable formats"""
    if df is None:
        return None
    
    cleaned_df = pd.DataFrame()
    
    for col in df.columns:
        col_data = df[col]
        col_dtype = str(col_data.dtype)
        
        # Skip problematic column types
        if 'duration' in col_dtype:
            try:
                cleaned_df[col] = col_data.dt.total_seconds()
            except:
                continue  # Skip this column
            continue
        
        # If column is already numeric, keep it
        if pd.api.types.is_numeric_dtype(col_data):
            cleaned_df[col] = col_data
            continue
        
        # If column is datetime, convert to string
        if pd.api.types.is_datetime64_any_dtype(col_data):
            cleaned_df[col] = col_data.astype(str)
            continue
        
        # For object columns, try to convert to numeric or string
        if col_data.dtype == 'object':
            # Try numeric conversion
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if numeric_data.notna().sum() > len(col_data) * 0.5:  # If >50% can be converted
                    cleaned_df[col] = numeric_data
                    continue
            except:
                pass
            
            # Convert to string
            cleaned_df[col] = col_data.astype(str)
            continue
        
        # For other types, convert to string
        try:
            cleaned_df[col] = col_data.astype(str)
        except:
            continue  # Skip problematic columns
    
    return cleaned_df

def download_file(filename):
    """Download a parquet file from GitHub"""
    if filename in _data_cache:
        return _data_cache[filename]
    
    url = f"{GITHUB_BASE_URL}/{filename}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        _data_cache[filename] = df
        return df
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def get_merged_data():
    """Get merged vehicle position + service alerts data"""
    return download_file(DATA_FILES["merged"])

def get_vehicle_data():
    """Get vehicle positions data"""
    return download_file(DATA_FILES["vehicle"])

def get_alerts_data():
    """Get service alerts data"""
    return download_file(DATA_FILES["alerts"])

def get_trip_updates_data():
    """Get trip updates data"""
    return download_file(DATA_FILES["trip_updates"])

def get_data_summary():
    """Get summary of available data"""
    summary = {}
    for key, filename in DATA_FILES.items():
        df = download_file(filename)
        if df is not None:
            summary[key] = {
                "rows": len(df),
                "columns": list(df.columns)
            }
    return summary
