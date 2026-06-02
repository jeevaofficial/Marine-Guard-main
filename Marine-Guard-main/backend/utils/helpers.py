"""
Helper Functions for Marine Safety System
==========================================
General utility functions used across the application.

Author: B.Tech AI&DS 
Date: 2026
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_safety_status(
    wave_height: float,
    wind_speed: Optional[float] = None
) -> Dict[str, Any]:
    """
    Classify marine safety status based on wave height and wind speed.
    
    Classification rules based on IMD and international maritime standards:
    - Safe: wave_height < 1.0 m
    - Caution: 1.0 m ≤ wave_height < 2.5 m
    - Dangerous: wave_height ≥ 2.5 m
    
    Wind speed can upgrade danger level:
    - If wind > 17 m/s and caution → Dangerous
    
    Args:
        wave_height: Predicted or current wave height in meters
        wind_speed: Optional wind speed in m/s
        
    Returns:
        Dictionary with status, color, and description
    """
    # Default classification based on wave height
    if wave_height < 1.0:
        status = "Safe"
        color = "#28a745"  # Green
        description = "Conditions are favorable for fishing and coastal activities."
        risk_level = 1
    elif wave_height < 2.5:
        status = "Caution"
        color = "#ffc107"  # Yellow/Amber
        description = "Exercise caution. Avoid venturing far from shore."
        risk_level = 2
    else:
        status = "Dangerous"
        color = "#dc3545"  # Red
        description = "Do not venture into sea. Stay away from coast."
        risk_level = 3
    
    # Upgrade to Dangerous if wind speed is very high
    if wind_speed is not None and wind_speed > 17.0 and risk_level < 3:
        status = "Dangerous"
        color = "#dc3545"
        description = "High wind speeds detected. Do not venture into sea."
        risk_level = 3
    
    return {
        "status": status,
        "color": color,
        "description": description,
        "risk_level": risk_level,
        "wave_height": round(wave_height, 2),
        "wind_speed": round(wind_speed, 1) if wind_speed else None
    }


def get_date_range_for_nasa(days_back: int = 30) -> Dict[str, str]:
    """
    Calculate date range for NASA POWER API historical data request.
    
    NASA POWER API has a delay of ~5 days for hourly data.
    This function calculates appropriate date range.
    
    Args:
        days_back: Number of days of historical data to fetch
        
    Returns:
        Dictionary with start and end dates in YYYYMMDD format
    """
    # NASA data has ~5 day lag
    end_date = datetime.now() - timedelta(days=5)
    start_date = end_date - timedelta(days=days_back)
    
    return {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "start_datetime": start_date,
        "end_datetime": end_date
    }


def format_coordinates(lat: float, lon: float, precision: int = 2) -> str:
    """
    Format coordinates for display.
    
    Args:
        lat: Latitude
        lon: Longitude
        precision: Decimal places
        
    Returns:
        Formatted string like "13.08°N, 80.27°E"
    """
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    
    return f"{abs(lat):.{precision}f}°{lat_dir}, {abs(lon):.{precision}f}°{lon_dir}"


def wind_direction_to_cardinal(degrees: float) -> str:
    """
    Convert wind direction in degrees to cardinal direction.
    
    Args:
        degrees: Wind direction in degrees (0-360)
        
    Returns:
        Cardinal direction string (N, NE, E, etc.)
    """
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    
    # Normalize to 0-360
    degrees = degrees % 360
    
    # Each direction covers 22.5 degrees
    index = round(degrees / 22.5) % 16
    
    return directions[index]


def meters_per_second_to_knots(mps: float) -> float:
    """Convert wind speed from m/s to knots."""
    return mps * 1.944


def meters_per_second_to_kmph(mps: float) -> float:
    """Convert wind speed from m/s to km/h."""
    return mps * 3.6


def beaufort_scale(wind_speed_mps: float) -> Dict[str, Any]:
    """
    Convert wind speed to Beaufort scale.
    
    Args:
        wind_speed_mps: Wind speed in meters per second
        
    Returns:
        Dictionary with Beaufort number, description, and sea state
    """
    # Beaufort scale thresholds in m/s
    beaufort_table = [
        (0.3, 0, "Calm", "Sea like mirror"),
        (1.6, 1, "Light air", "Ripples without crests"),
        (3.4, 2, "Light breeze", "Small wavelets"),
        (5.5, 3, "Gentle breeze", "Large wavelets"),
        (8.0, 4, "Moderate breeze", "Small waves"),
        (10.8, 5, "Fresh breeze", "Moderate waves"),
        (13.9, 6, "Strong breeze", "Large waves"),
        (17.2, 7, "Near gale", "Sea heaps up"),
        (20.8, 8, "Gale", "Moderately high waves"),
        (24.5, 9, "Severe gale", "High waves"),
        (28.5, 10, "Storm", "Very high waves"),
        (32.7, 11, "Violent storm", "Exceptionally high waves"),
        (float('inf'), 12, "Hurricane", "Air filled with foam and spray")
    ]
    
    for threshold, number, description, sea_state in beaufort_table:
        if wind_speed_mps < threshold:
            return {
                "number": number,
                "description": description,
                "sea_state": sea_state,
                "wind_speed_mps": round(wind_speed_mps, 1)
            }
    
    return beaufort_table[-1]


def create_forecast_summary(
    hourly_predictions: List[float],
    timestamps: List[datetime]
) -> Dict[str, Any]:
    """
    Create a summary of hourly forecast predictions.
    
    Args:
        hourly_predictions: List of predicted values
        timestamps: Corresponding timestamps
        
    Returns:
        Dictionary with forecast summary statistics
    """
    import numpy as np
    
    predictions = np.array(hourly_predictions)
    
    summary = {
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "count": len(predictions),
        "trend": "increasing" if predictions[-1] > predictions[0] else "decreasing",
    }
    
    # Find peak time
    peak_idx = np.argmax(predictions)
    summary["peak_value"] = float(predictions[peak_idx])
    summary["peak_time"] = timestamps[peak_idx].isoformat() if timestamps else None
    
    # Find safest time (lowest wave height)
    safest_idx = np.argmin(predictions)
    summary["safest_value"] = float(predictions[safest_idx])
    summary["safest_time"] = timestamps[safest_idx].isoformat() if timestamps else None
    
    return summary


def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def load_json_file(path: str) -> Dict:
    """Load JSON file safely."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {path}: {e}")
        return {}


def save_json_file(data: Dict, path: str) -> bool:
    """Save data to JSON file."""
    try:
        ensure_directory_exists(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved JSON to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        return False


def format_timestamp(dt: datetime, format_type: str = "display") -> str:
    """
    Format datetime for different uses.
    
    Args:
        dt: Datetime object
        format_type: One of 'display', 'api', 'filename'
        
    Returns:
        Formatted string
    """
    formats = {
        "display": "%d %b %Y, %I:%M %p",  # "02 Feb 2026, 03:30 PM"
        "api": "%Y-%m-%dT%H:%M:%S",       # ISO format
        "filename": "%Y%m%d_%H%M%S",      # For file names
        "date_only": "%Y-%m-%d"
    }
    
    return dt.strftime(formats.get(format_type, formats["display"]))


def calculate_forecast_confidence(
    historical_rmse: float,
    forecast_horizon: int,
    max_horizon: int = 24
) -> float:
    """
    Calculate confidence score for forecast.
    
    Confidence decreases with longer forecast horizons.
    
    Args:
        historical_rmse: Root Mean Square Error from model validation
        forecast_horizon: Hours ahead being predicted
        max_horizon: Maximum forecast horizon
        
    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from RMSE (lower is better)
    # Assuming typical wave heights of 0-5m, RMSE of 0.3 is good
    rmse_confidence = max(0, 1 - (historical_rmse / 2))
    
    # Time decay factor
    time_decay = 1 - (forecast_horizon / (max_horizon * 2))
    
    # Combined confidence
    confidence = rmse_confidence * time_decay
    
    return round(max(0, min(1, confidence)), 2)
