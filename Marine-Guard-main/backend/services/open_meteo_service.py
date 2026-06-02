"""
Open-Meteo Marine API Service
==============================
This module handles all interactions with Open-Meteo Marine API for wave forecasts.

Open-Meteo Marine API provides:
- Real-time marine conditions
- 7-day wave forecasts
- Free access, no API key required
- Hourly temporal resolution

API Documentation: https://open-meteo.com/en/docs/marine-weather-api

Author: B.Tech AI&DS 
Date: 2026
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    OPEN_METEO_MARINE_URL, 
    OPEN_METEO_PARAMETERS,
    OPEN_METEO_WEATHER_URL,
    OPEN_METEO_WEATHER_PARAMETERS,
    COASTAL_DISTRICTS
)

# Set up logging
logger = logging.getLogger(__name__)


class OpenMeteoMarineService:
    """
    Service class for Open-Meteo Marine API interactions.
    
    This class provides methods to:
    1. Fetch current and forecasted marine conditions
    2. Get wave height, period, and direction forecasts
    3. Combine with weather data for comprehensive forecasts
    4. Handle API errors gracefully
    """
    
    def __init__(self):
        """Initialize Open-Meteo Marine service."""
        self.marine_url = OPEN_METEO_MARINE_URL
        self.weather_url = OPEN_METEO_WEATHER_URL
        self.marine_parameters = OPEN_METEO_PARAMETERS
        self.weather_parameters = OPEN_METEO_WEATHER_PARAMETERS
        
        logger.info("Open-Meteo Marine Service initialized")
    
    def fetch_marine_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7
    ) -> pd.DataFrame:
        """
        Fetch marine forecast data from Open-Meteo API.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            forecast_days: Number of days to forecast (1-7)
            
        Returns:
            DataFrame with hourly marine forecast data
        
        Example:
            >>> service = OpenMeteoMarineService()
            >>> df = service.fetch_marine_forecast(13.08, 80.27, forecast_days=3)
        """
        # Build API URL with parameters
        params_string = ",".join(self.marine_parameters)
        url = (
            f"{self.marine_url}"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            f"&hourly={params_string}"
            f"&forecast_days={forecast_days}"
            f"&timezone=Asia/Kolkata"
        )
        
        logger.info(f"Fetching marine forecast for ({latitude}, {longitude})")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Open-Meteo Marine API error: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                raise Exception(f"Open-Meteo Marine API returned status {response.status_code}")
            
            data = response.json()
            
            # Parse response
            df = self._parse_marine_response(data)
            
            logger.info(f"Successfully fetched {len(df)} hourly marine forecast records")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def _parse_marine_response(self, data: Dict) -> pd.DataFrame:
        """
        Parse Open-Meteo Marine API response into DataFrame.
        
        Args:
            data: JSON response from API
            
        Returns:
            DataFrame with timestamp index and forecast columns
        """
        try:
            hourly_data = data.get("hourly", {})
            
            if not hourly_data or "time" not in hourly_data:
                raise ValueError("No hourly data in response")
            
            # Extract time and parameters
            df = pd.DataFrame(hourly_data)
            
            # Parse timestamps
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Interpolate missing values (now safe since all columns are numeric)
            df = df.interpolate(method='linear', limit=2)
            
            # Drop any rows still containing NaN after interpolation
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing marine response: {e}")
            raise
    
    def fetch_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7
    ) -> pd.DataFrame:
        """
        Fetch weather forecast data from Open-Meteo Weather API.
        
        This complements the marine data with atmospheric conditions.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with hourly weather forecast
        """
        params_string = ",".join(self.weather_parameters)
        url = (
            f"{self.weather_url}"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            f"&hourly={params_string}"
            f"&forecast_days={forecast_days}"
            f"&wind_speed_unit=ms"
            f"&timezone=Asia/Kolkata"
        )
        
        logger.info(f"Fetching weather forecast for ({latitude}, {longitude})")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Open-Meteo Weather API error: {response.status_code}")
                raise Exception(f"Open-Meteo Weather API returned status {response.status_code}")
            
            data = response.json()
            hourly_data = data.get("hourly", {})
            
            if not hourly_data:
                raise ValueError("No hourly data in weather response")
            
            df = pd.DataFrame(hourly_data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Successfully fetched {len(df)} hourly weather forecast records")
            return df
            
        except Exception as e:
            logger.error(f"Weather forecast request failed: {e}")
            raise
    
    def fetch_combined_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7
    ) -> pd.DataFrame:
        """
        Fetch combined marine and weather forecast.
        
        This method merges marine (wave) data with atmospheric data
        for comprehensive forecasting.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            forecast_days: Number of days to forecast
            
        Returns:
            Combined DataFrame with both marine and weather data
        """
        # Fetch both datasets
        marine_df = self.fetch_marine_forecast(latitude, longitude, forecast_days)
        weather_df = self.fetch_weather_forecast(latitude, longitude, forecast_days)
        
        # Merge on index (timestamp)
        combined_df = pd.merge(
            marine_df,
            weather_df,
            left_index=True,
            right_index=True,
            how='outer'
        )
        
        # Ensure all columns are numeric
        for col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Interpolate any gaps
        combined_df = combined_df.interpolate(method='linear', limit=1)
        combined_df = combined_df.dropna()
        
        logger.info(f"Combined forecast shape: {combined_df.shape}")
        return combined_df
    
    def fetch_for_district(
        self,
        district_name: str,
        forecast_days: int = 3,
        include_weather: bool = True
    ) -> pd.DataFrame:
        """
        Fetch forecast data for a Tamil Nadu coastal district.
        
        Args:
            district_name: Name of the coastal district
            forecast_days: Number of days to forecast
            include_weather: Whether to include weather data
            
        Returns:
            DataFrame with forecast data for the district
        """
        if district_name not in COASTAL_DISTRICTS:
            available = ", ".join(COASTAL_DISTRICTS.keys())
            raise ValueError(f"Unknown district: {district_name}. Available: {available}")
        
        coords = COASTAL_DISTRICTS[district_name]
        
        logger.info(f"Fetching forecast for {district_name} ({coords['lat']}, {coords['lon']})")
        
        if include_weather:
            return self.fetch_combined_forecast(
                coords['lat'], 
                coords['lon'], 
                forecast_days
            )
        else:
            return self.fetch_marine_forecast(
                coords['lat'], 
                coords['lon'], 
                forecast_days
            )
    
    def get_current_conditions(
        self,
        district_name: str
    ) -> Dict:
        """
        Get current marine conditions for a district.
        
        Args:
            district_name: Name of the coastal district
            
        Returns:
            Dictionary with current conditions
        """
        df = self.fetch_for_district(district_name, forecast_days=1)
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning(f"No marine data available for {district_name}")
            return {
                "timestamp": datetime.now().isoformat(),
                "district": district_name,
                "coordinates": COASTAL_DISTRICTS[district_name],
                "wave_height": 0.0,
                "wave_period": 0.0,
                "wave_direction": 0,
                "wind_speed": 0.0,
                "wind_direction": 0,
                "temperature": 25.0,
                "humidity": 70,
                "pressure": 1013.0,
                "data_unavailable": True
            }
        
        # Get the closest timestamp to now
        now = datetime.now()
        closest_idx = df.index.get_indexer([now], method='nearest')[0]
        current = df.iloc[closest_idx]
        
        # Build response dictionary
        conditions = {
            "timestamp": str(df.index[closest_idx]),
            "district": district_name,
            "coordinates": COASTAL_DISTRICTS[district_name],
            "wave_height": round(float(current.get("wave_height", 0)), 2),
            "wave_period": round(float(current.get("wave_period", 0)), 1),
            "wave_direction": round(float(current.get("wave_direction", 0)), 0),
        }
        
        # Add weather data if available
        if "wind_speed_10m" in current:
            conditions["wind_speed"] = round(float(current.get("wind_speed_10m", 0)), 1)
            conditions["wind_direction"] = round(float(current.get("wind_direction_10m", 0)), 0)
            conditions["temperature"] = round(float(current.get("temperature_2m", 25)), 1)
            conditions["humidity"] = round(float(current.get("relative_humidity_2m", 70)), 0)
            conditions["pressure"] = round(float(current.get("surface_pressure", 1013)), 1)
        
        return conditions
    
    def get_forecast_summary(
        self,
        district_name: str,
        hours_ahead: int = 24
    ) -> Dict:
        """
        Get forecast summary for a district.
        
        Args:
            district_name: Name of the coastal district
            hours_ahead: Number of hours to include in summary
            
        Returns:
            Dictionary with forecast statistics
        """
        df = self.fetch_for_district(district_name, forecast_days=3)
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning(f"No forecast data available for {district_name}")
            return {
                "district": district_name,
                "forecast_hours": hours_ahead,
                "wave_height": {
                    "current": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                },
                "hourly_wave_heights": [],
                "timestamps": [],
                "data_unavailable": True
            }
        
        # Filter to requested forecast horizon
        now = datetime.now()
        end_time = now + timedelta(hours=hours_ahead)
        
        # Get the closest timestamp to now as starting point
        closest_idx = df.index.get_indexer([now], method='nearest')[0]
        
        # Filter from current time onwards for the forecast period
        forecast_df = df.iloc[closest_idx:]
        forecast_df = forecast_df[forecast_df.index <= end_time]
        
        # Handle empty after filtering
        if forecast_df.empty:
            forecast_df = df.head(hours_ahead)
        
        summary = {
            "district": district_name,
            "forecast_hours": hours_ahead,
            "wave_height": {
                "current": round(float(forecast_df["wave_height"].iloc[0]), 2),
                "min": round(float(forecast_df["wave_height"].min()), 2),
                "max": round(float(forecast_df["wave_height"].max()), 2),
                "mean": round(float(forecast_df["wave_height"].mean()), 2),
            },
            "hourly_wave_heights": forecast_df["wave_height"].round(2).tolist(),
            "timestamps": [t.isoformat() for t in forecast_df.index],
        }
        
        # Add wind data if available
        if "wind_speed_10m" in forecast_df.columns:
            summary["wind_speed"] = {
                "current": round(float(forecast_df["wind_speed_10m"].iloc[0]), 1),
                "max": round(float(forecast_df["wind_speed_10m"].max()), 1),
                "mean": round(float(forecast_df["wind_speed_10m"].mean()), 1),
            }
        
        # Add hourly data for detailed analysis (wind rose, etc.)
        hourly_data = []
        for idx, row in forecast_df.iterrows():
            hourly_record = {
                "timestamp": idx.isoformat(),
                "wave_height": round(float(row["wave_height"]), 2),
            }
            
            if "wind_speed_10m" in row:
                hourly_record["wind_speed"] = round(float(row["wind_speed_10m"]), 1)
            
            if "wind_direction_10m" in row:
                hourly_record["wind_direction"] = round(float(row["wind_direction_10m"]), 1)
            
            if "wave_direction" in row:
                hourly_record["wave_direction"] = round(float(row["wave_direction"]), 1)
            
            if "wave_period" in row:
                hourly_record["wave_period"] = round(float(row["wave_period"]), 1)
            
            hourly_data.append(hourly_record)
        
        summary["hourly_data"] = hourly_data
        
        return summary
    
    def fetch_all_districts_current(self) -> Dict[str, Dict]:
        """
        Fetch current conditions for all 14 coastal districts.
        
        Returns:
            Dictionary mapping district names to current conditions
        """
        results = {}
        
        for district_name in COASTAL_DISTRICTS.keys():
            try:
                conditions = self.get_current_conditions(district_name)
                results[district_name] = conditions
            except Exception as e:
                logger.error(f"Failed to fetch conditions for {district_name}: {e}")
                results[district_name] = {"error": str(e)}
            
            # Small delay to be nice to the API
            time.sleep(0.5)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = OpenMeteoMarineService()
    
    # Test fetching data for Chennai
    print("\nFetching Open-Meteo marine forecast for Chennai...")
    try:
        df = service.fetch_for_district("Chennai", forecast_days=2)
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:\n{df.head()}")
        
        # Test current conditions
        print("\nCurrent conditions:")
        current = service.get_current_conditions("Chennai")
        for key, value in current.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
