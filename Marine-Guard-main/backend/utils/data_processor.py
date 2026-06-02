"""
Data Processing Utilities for Marine Safety System
===================================================
This module provides functions for:
1. Data cleaning and validation
2. Timestamp alignment between different data sources
3. Feature engineering for model inputs
4. Sequence preparation for GRU models
5. Data normalization and scaling

Author: B.Tech AI&DS 
Date: 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Set up logging
logger = logging.getLogger(__name__)


def clean_nasa_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess NASA POWER API data.
    
    Operations performed:
    1. Convert timestamp to datetime index
    2. Handle missing values (interpolation for small gaps)
    3. Remove outliers using IQR method
    4. Ensure proper data types
    
    Args:
        df: Raw DataFrame from NASA POWER API
        
    Returns:
        Cleaned DataFrame with datetime index
    """
    logger.info("Cleaning NASA POWER data...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Ensure datetime index
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        df_clean.set_index('timestamp', inplace=True)
    
    # Handle missing values
    # NASA POWER uses -999 as missing value indicator
    df_clean = df_clean.replace(-999, np.nan)
    df_clean = df_clean.replace(-999.0, np.nan)
    
    # Interpolate small gaps (up to 3 hours)
    df_clean = df_clean.interpolate(method='time', limit=3)
    
    # Forward fill remaining small gaps
    df_clean = df_clean.fillna(method='ffill', limit=2)
    
    # Backward fill any remaining
    df_clean = df_clean.fillna(method='bfill', limit=2)
    
    # Remove any rows with remaining NaN values
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    removed = initial_len - len(df_clean)
    
    if removed > 0:
        logger.warning(f"Removed {removed} rows with missing values")
    
    # Remove outliers using IQR method
    for column in df_clean.columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Cap outliers instead of removing
        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
    
    logger.info(f"NASA data cleaned. Shape: {df_clean.shape}")
    return df_clean


def clean_marine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess Open-Meteo Marine API data.
    
    Operations performed:
    1. Parse timestamp column
    2. Handle missing values
    3. Validate physical ranges
    4. Sort by timestamp
    
    Args:
        df: Raw DataFrame from Open-Meteo Marine API
        
    Returns:
        Cleaned DataFrame with datetime index
    """
    logger.info("Cleaning marine data...")
    
    df_clean = df.copy()
    
    # Ensure datetime index
    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_datetime(df_clean['time'])
        df_clean.set_index('time', inplace=True)
    elif 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        df_clean.set_index('timestamp', inplace=True)
    
    # Handle missing values
    df_clean = df_clean.interpolate(method='time', limit=2)
    df_clean = df_clean.fillna(method='ffill', limit=1)
    df_clean = df_clean.fillna(method='bfill', limit=1)
    
    # Validate physical ranges for wave height (must be non-negative)
    if 'wave_height' in df_clean.columns:
        df_clean['wave_height'] = df_clean['wave_height'].clip(lower=0)
    
    # Validate wind direction (0-360 degrees)
    if 'wind_direction' in df_clean.columns:
        df_clean['wind_direction'] = df_clean['wind_direction'] % 360
    
    # Sort by timestamp
    df_clean = df_clean.sort_index()
    
    logger.info(f"Marine data cleaned. Shape: {df_clean.shape}")
    return df_clean


def align_timestamps(
    climate_df: pd.DataFrame,
    marine_df: pd.DataFrame,
    freq: str = 'H'
) -> pd.DataFrame:
    """
    Align climate and marine data by timestamp.
    
    This function merges data from NASA POWER (climate) and Open-Meteo (marine)
    APIs by matching their timestamps. This is crucial because:
    - Both datasets may have different time resolutions
    - Predictions need synchronized features
    
    Args:
        climate_df: DataFrame with climate data (NASA POWER)
        marine_df: DataFrame with marine data (Open-Meteo)
        freq: Frequency for resampling ('H' for hourly)
        
    Returns:
        Merged DataFrame with aligned timestamps
    """
    logger.info("Aligning timestamps between climate and marine data...")
    
    # Resample both to same frequency
    climate_resampled = climate_df.resample(freq).mean()
    marine_resampled = marine_df.resample(freq).mean()
    
    # Find common time range
    start_time = max(climate_resampled.index.min(), marine_resampled.index.min())
    end_time = min(climate_resampled.index.max(), marine_resampled.index.max())
    
    logger.info(f"Common time range: {start_time} to {end_time}")
    
    # Filter to common range
    climate_filtered = climate_resampled[start_time:end_time]
    marine_filtered = marine_resampled[start_time:end_time]
    
    # Merge on index
    merged_df = pd.merge(
        climate_filtered,
        marine_filtered,
        left_index=True,
        right_index=True,
        how='inner',
        suffixes=('_climate', '_marine')
    )
    
    # Interpolate any small gaps created by merge
    merged_df = merged_df.interpolate(method='time', limit=1)
    merged_df = merged_df.dropna()
    
    logger.info(f"Aligned data shape: {merged_df.shape}")
    return merged_df


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int,
    target_column_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting with GRU.
    
    This function prepares data in the format required by GRU networks:
    - Input (X): Sequences of historical observations
    - Output (y): Future values to predict
    
    Example with sequence_length=3, forecast_horizon=2:
    Data: [1, 2, 3, 4, 5, 6, 7]
    X[0]: [1, 2, 3] → y[0]: [4, 5]
    X[1]: [2, 3, 4] → y[1]: [5, 6]
    
    Args:
        data: 2D numpy array with shape (timesteps, features)
        sequence_length: Number of past timesteps to use as input
        forecast_horizon: Number of future timesteps to predict
        target_column_idx: Index of the column to predict
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    X, y = [], []
    
    total_len = len(data)
    
    for i in range(total_len - sequence_length - forecast_horizon + 1):
        # Input sequence: all features for sequence_length timesteps
        X.append(data[i:(i + sequence_length), :])
        
        # Output: target column for forecast_horizon timesteps
        y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), target_column_idx])
    
    return np.array(X), np.array(y)


def create_sequences_multioutput(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int,
    target_columns: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multi-output forecasting.
    
    Similar to create_sequences but predicts multiple target columns.
    Useful when we need to forecast multiple climate variables simultaneously.
    
    Args:
        data: 2D numpy array with shape (timesteps, features)
        sequence_length: Number of past timesteps to use as input
        forecast_horizon: Number of future timesteps to predict
        target_columns: List of column indices to predict
        
    Returns:
        Tuple of (X, y) arrays where y has shape (samples, horizon, num_targets)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), target_columns])
    
    return np.array(X), np.array(y)


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare feature matrix for model training.
    
    Args:
        df: DataFrame with all columns
        feature_columns: List of column names to use as features
        target_column: Name of the target column
        
    Returns:
        Tuple of (feature_matrix, column_order)
    """
    # Ensure target is first column (index 0)
    columns = [target_column] + [c for c in feature_columns if c != target_column]
    
    # Extract and convert to numpy
    data = df[columns].values
    
    return data, columns


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features for temporal patterns.
    
    Adds sine/cosine encoded features for:
    - Hour of day (captures daily patterns)
    - Day of week (captures weekly patterns)
    - Month (captures seasonal patterns)
    
    Cyclical encoding prevents the model from seeing
    11 PM and 12 AM as very different times.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added time features
    """
    df_feat = df.copy()
    
    # Hour of day (0-23) → cyclical
    hour = df_feat.index.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0-6) → cyclical
    day = df_feat.index.dayofweek
    df_feat['day_sin'] = np.sin(2 * np.pi * day / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * day / 7)
    
    # Month (1-12) → cyclical (for seasonal patterns)
    month = df_feat.index.month
    df_feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    return df_feat


def normalize_data(
    data: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMax scaling.
    
    Scaling is essential for neural networks because:
    1. Features may have different ranges
    2. Gradients flow better with normalized data
    3. Training converges faster
    
    Args:
        data: 2D numpy array to normalize
        scaler: Pre-fitted scaler (for inference)
        fit: Whether to fit the scaler on this data
        
    Returns:
        Tuple of (normalized_data, fitted_scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    if fit:
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)
    
    return normalized, scaler


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: MinMaxScaler,
    target_idx: int = 0,
    num_features: int = None
) -> np.ndarray:
    """
    Inverse transform predictions back to original scale.
    
    Args:
        predictions: Normalized predictions from model
        scaler: Fitted MinMaxScaler used during training
        target_idx: Index of the target feature
        num_features: Total number of features
        
    Returns:
        Predictions in original scale
    """
    if num_features is None:
        num_features = len(scaler.min_)
    
    # Create dummy array with same shape as training data
    dummy = np.zeros((len(predictions), num_features))
    
    # Handle different prediction shapes
    if predictions.ndim == 1:
        dummy[:, target_idx] = predictions
    elif predictions.ndim == 2:
        # Multi-step predictions - take first column or flatten
        if predictions.shape[1] == 1:
            dummy[:, target_idx] = predictions.flatten()
        else:
            # Average across forecast horizon
            dummy[:, target_idx] = predictions.mean(axis=1)
    
    # Inverse transform
    inverse = scaler.inverse_transform(dummy)
    
    return inverse[:, target_idx]


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """Save fitted scaler to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path: str) -> MinMaxScaler:
    """Load fitted scaler from disk."""
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded from {path}")
    return scaler


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features that may improve predictions.
    
    New features:
    - Wave steepness (wave_height / wave_period)
    - Wind chill factor
    - Pressure change rate
    - Temperature-humidity index
    
    Args:
        df: DataFrame with basic features
        
    Returns:
        DataFrame with additional derived features
    """
    df_derived = df.copy()
    
    # Wave steepness (if marine data available)
    if 'wave_height' in df.columns and 'wave_period' in df.columns:
        df_derived['wave_steepness'] = df_derived['wave_height'] / (df_derived['wave_period'] + 0.1)
    
    # Pressure change (rate of change)
    if 'PS' in df.columns:
        df_derived['pressure_change'] = df_derived['PS'].diff().fillna(0)
    elif 'surface_pressure' in df.columns:
        df_derived['pressure_change'] = df_derived['surface_pressure'].diff().fillna(0)
    
    # Temperature-humidity index (feels-like indicator)
    if 'T2M' in df.columns and 'RH2M' in df.columns:
        # Simplified heat index formula
        T = df_derived['T2M']
        RH = df_derived['RH2M']
        df_derived['heat_index'] = T - ((100 - RH) / 5)
    
    return df_derived


def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    """
    Validate data quality and return statistics.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "missing_columns": [],
        "null_percentages": {},
        "data_range": {},
        "row_count": len(df),
        "time_range": {}
    }
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            results["missing_columns"].append(col)
            results["valid"] = False
    
    if not results["valid"]:
        return results
    
    # Calculate null percentages
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        results["null_percentages"][col] = round(null_pct, 2)
        
        if null_pct > 20:
            logger.warning(f"Column {col} has {null_pct:.1f}% missing values")
    
    # Data ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        results["data_range"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    
    # Time range
    if isinstance(df.index, pd.DatetimeIndex):
        results["time_range"] = {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "duration_hours": (df.index.max() - df.index.min()).total_seconds() / 3600
        }
    
    return results
