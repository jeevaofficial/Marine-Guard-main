"""
NASA POWER API Service
======================
This module handles all interactions with NASA POWER API for historical climate data.

NASA POWER (Prediction of Worldwide Energy Resource) provides:
- Historical meteorological data
- Free access, no API key required
- Hourly temporal resolution
- Global coverage

API Documentation: https://power.larc.nasa.gov/docs/services/api/

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
from config.settings import NASA_POWER_BASE_URL, NASA_POWER_PARAMETERS, COASTAL_DISTRICTS

# Set up logging
logger = logging.getLogger(__name__)


class NASAPowerService:
    """
    Service class for NASA POWER API interactions.
    
    This class provides methods to:
    1. Fetch historical climate data for any location
    2. Process and clean the raw API response
    3. Handle API rate limits and errors
    4. Cache responses to reduce API calls
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize NASA POWER service.
        
        Args:
            cache_dir: Directory to cache API responses (optional)
        """
        self.base_url = NASA_POWER_BASE_URL
        self.parameters = NASA_POWER_PARAMETERS
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        logger.info("NASA POWER Service initialized")
    
    def fetch_historical_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch historical climate data from NASA POWER API.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            start_date: Start date in format 'YYYYMMDD'
            end_date: End date in format 'YYYYMMDD'
            parameters: List of parameters to fetch (default: all configured)
            
        Returns:
            DataFrame with hourly climate data
        
        Example:
            >>> service = NASAPowerService()
            >>> df = service.fetch_historical_data(13.08, 80.27, '20250101', '20250131')
        """
        if parameters is None:
            parameters = self.parameters
        
        # Build API URL
        params_string = ",".join(parameters)
        url = (
            f"{self.base_url}"
            f"?start={start_date}"
            f"&end={end_date}"
            f"&latitude={latitude}"
            f"&longitude={longitude}"
            f"&community=RE"  # Renewable Energy community has best coverage
            f"&parameters={params_string}"
            f"&format=JSON"
            f"&time-standard=UTC"
        )
        
        logger.info(f"Fetching NASA POWER data for ({latitude}, {longitude}) from {start_date} to {end_date}")
        
        try:
            # Make API request with retry logic
            response = self._make_request_with_retry(url)
            
            if response.status_code != 200:
                logger.error(f"NASA POWER API error: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                raise Exception(f"NASA POWER API returned status {response.status_code}")
            
            data = response.json()
            
            # Parse the response into DataFrame
            df = self._parse_response(data, parameters)
            
            logger.info(f"Successfully fetched {len(df)} hourly records from NASA POWER")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def _make_request_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> requests.Response:
        """
        Make HTTP request with retry logic for handling rate limits.
        
        Args:
            url: Full API URL
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds to wait between retries
            
        Returns:
            Response object
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                
                # Check for rate limiting (429)
                if response.status_code == 429:
                    logger.warning(f"Rate limited. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
        raise Exception("Max retries exceeded for NASA POWER API")
    
    def _parse_response(
        self,
        data: Dict,
        parameters: List[str]
    ) -> pd.DataFrame:
        """
        Parse NASA POWER API JSON response into DataFrame.
        
        The API returns data in a nested structure:
        {
            "properties": {
                "parameter": {
                    "T2M": {"2025010100": 25.5, "2025010101": 25.3, ...},
                    "RH2M": {...},
                    ...
                }
            }
        }
        
        Args:
            data: JSON response from API
            parameters: List of parameters requested
            
        Returns:
            DataFrame with timestamp index and parameter columns
        """
        try:
            param_data = data.get("properties", {}).get("parameter", {})
            
            if not param_data:
                logger.error("No parameter data in response")
                raise ValueError("Empty response from NASA POWER API")
            
            # Create dictionary for DataFrame
            records = {}
            
            for param in parameters:
                if param in param_data:
                    records[param] = param_data[param]
                else:
                    logger.warning(f"Parameter {param} not found in response")
            
            if not records:
                raise ValueError("No valid parameters found in response")
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Parse timestamps (format: YYYYMMDDHH)
            df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
            df.index.name = 'timestamp'
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Replace missing value indicators (-999)
            df = df.replace(-999, np.nan)
            df = df.replace(-999.0, np.nan)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing NASA POWER response: {e}")
            raise
    
    def fetch_for_district(
        self,
        district_name: str,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical climate data for a Tamil Nadu coastal district.
        
        This is a convenience method that looks up district coordinates
        and calculates appropriate date ranges automatically.
        
        Args:
            district_name: Name of the coastal district
            days_back: Number of days of historical data to fetch
            
        Returns:
            DataFrame with climate data for the district
        """
        if district_name not in COASTAL_DISTRICTS:
            available = ", ".join(COASTAL_DISTRICTS.keys())
            raise ValueError(f"Unknown district: {district_name}. Available: {available}")
        
        coords = COASTAL_DISTRICTS[district_name]
        
        # Calculate date range (NASA has ~5 day lag)
        end_date = datetime.now() - timedelta(days=5)
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching data for {district_name} ({coords['lat']}, {coords['lon']})")
        
        return self.fetch_historical_data(
            latitude=coords['lat'],
            longitude=coords['lon'],
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
    
    def fetch_all_districts(
        self,
        days_back: int = 30,
        delay_between_requests: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch climate data for all 14 coastal districts.
        
        Includes delay between requests to avoid rate limiting.
        
        Args:
            days_back: Number of days of historical data
            delay_between_requests: Seconds to wait between API calls
            
        Returns:
            Dictionary mapping district names to DataFrames
        """
        results = {}
        
        for i, district_name in enumerate(COASTAL_DISTRICTS.keys()):
            logger.info(f"Fetching {i+1}/14: {district_name}")
            
            try:
                df = self.fetch_for_district(district_name, days_back)
                results[district_name] = df
                
                # Delay to avoid rate limiting (except for last request)
                if i < len(COASTAL_DISTRICTS) - 1:
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {district_name}: {e}")
                results[district_name] = None
        
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Successfully fetched data for {successful}/14 districts")
        
        return results
    
    def get_training_data(
        self,
        district_name: str,
        days_back: int = 60
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Get preprocessed training data for a district.
        
        This method:
        1. Fetches raw data from NASA POWER
        2. Cleans and preprocesses the data
        3. Adds derived features
        4. Returns train-ready DataFrame with metadata
        
        Args:
            district_name: Name of the coastal district
            days_back: Days of historical data for training
            
        Returns:
            Tuple of (processed_DataFrame, metadata_dict)
        """
        # Fetch raw data
        df = self.fetch_for_district(district_name, days_back)
        
        # Import data processor
        from utils.data_processor import clean_nasa_data, add_time_features, calculate_derived_features
        
        # Clean data
        df_clean = clean_nasa_data(df)
        
        # Add time features
        df_features = add_time_features(df_clean)
        
        # Add derived features
        df_final = calculate_derived_features(df_features)
        
        # Prepare metadata
        metadata = {
            "district": district_name,
            "coordinates": COASTAL_DISTRICTS[district_name],
            "date_range": {
                "start": str(df_final.index.min()),
                "end": str(df_final.index.max())
            },
            "num_records": len(df_final),
            "features": list(df_final.columns),
            "missing_percentage": df_final.isnull().sum().sum() / df_final.size * 100
        }
        
        logger.info(f"Training data prepared for {district_name}: {len(df_final)} records, {len(df_final.columns)} features")
        
        return df_final, metadata


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = NASAPowerService()
    
    # Test fetching data for Chennai
    print("\nFetching NASA POWER data for Chennai...")
    try:
        df = service.fetch_for_district("Chennai", days_back=7)
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nData statistics:\n{df.describe()}")
    except Exception as e:
        print(f"Error: {e}")
