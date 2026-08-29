"""
Configuration Settings for Marine Safety Forecasting System
============================================================
This module contains all configuration parameters for the system including:
- API endpoints (NASA POWER, Open-Meteo Marine)
- District coordinates for Tamil Nadu coastal region
- Model hyperparameters for GRU networks
- Safety thresholds for wave classification

Author: B.Tech AI&DS Project
Date: 2026
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# COASTAL DISTRICTS OF TAMIL NADU
# =============================================================================
# These are all 14 coastal districts with their approximate coastal coordinates
# Coordinates represent points along the coastline for accurate marine data

COASTAL_DISTRICTS = {
    "Thiruvallur": {"lat": 13.35, "lon": 80.30, "description": "Northern coastal district"},
    "Chennai": {"lat": 13.08, "lon": 80.35, "description": "State capital, major port city"},
    "Kanchipuram": {"lat": 12.68, "lon": 80.20, "description": "Ancient temple town with coast"},
    "Chengalpattu": {"lat": 12.41, "lon": 80.20, "description": "Coastal district south of Chennai"},
    "Villupuram": {"lat": 11.93, "lon": 79.90, "description": "Agricultural coastal district"},
    "Cuddalore": {"lat": 11.75, "lon": 79.85, "description": "Major fishing harbor"},
    "Mayiladuthurai": {"lat": 11.10, "lon": 79.95, "description": "Cauvery delta coastal region"},
    "Nagapattinam": {"lat": 10.77, "lon": 79.90, "description": "Historic port, fishing community"},
    "Thanjavur": {"lat": 10.60, "lon": 79.90, "description": "Delta region with coastal access"},
    "Tiruvarur": {"lat": 10.55, "lon": 79.95, "description": "Coastal delta district"},
    "Ramanathapuram": {"lat": 9.38, "lon": 79.00, "description": "Gulf of Mannar, Pamban region"},
    "Thoothukudi": {"lat": 8.78, "lon": 78.25, "description": "Major port, pearl fishing"},
    "Tirunelveli": {"lat": 8.50, "lon": 78.10, "description": "Southern coastal stretch"},
    "Kanniyakumari": {"lat": 8.08, "lon": 77.60, "description": "Southernmost tip of India"},
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

# NASA POWER API Configuration (FREE, NO KEY REQUIRED)
# Documentation: https://power.larc.nasa.gov/docs/services/api/
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA POWER parameters we need for climate forecasting
NASA_POWER_PARAMETERS = [
    "T2M",          # Temperature at 2 meters (°C)
    "RH2M",         # Relative Humidity at 2 meters (%)
    "WS2M",         # Wind Speed at 2 meters (m/s)
    "PS",           # Surface Pressure (kPa)
    "PRECTOTCORR",  # Precipitation Corrected (mm/hour)
]

# Open-Meteo Marine API Configuration (FREE, NO KEY REQUIRED)
# Documentation: https://open-meteo.com/en/docs/marine-weather-api
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Open-Meteo parameters for marine conditions
OPEN_METEO_PARAMETERS = [
    "wave_height",           # Significant wave height (m)
    "wave_period",           # Wave period (seconds)
    "wave_direction",        # Wave direction (degrees)
    "wind_wave_height",      # Wind wave height (m)
    "swell_wave_height",     # Swell wave height (m)
]

# Open-Meteo Weather API for additional forecasts
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_WEATHER_PARAMETERS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
]

# =============================================================================
# GROQ AI CONFIGURATION
# =============================================================================
# These values should be set in .env file for security

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# =============================================================================
# MODEL HYPERPARAMETERS (OPTIMIZED FOR CPU TRAINING)
# =============================================================================

# GRU Climate Forecasting Model Configuration
CLIMATE_MODEL_CONFIG = {
    "sequence_length": 24,          # 24 hours of historical data as input
    "forecast_horizon": 12,         # Predict next 12 hours
    "gru_units_layer1": 32,         # First GRU layer units
    "gru_units_layer2": 16,         # Second GRU layer units
    "dropout_rate": 0.2,            # Dropout for regularization
    "learning_rate": 0.001,         # Adam optimizer learning rate
    "batch_size": 32,               # Training batch size
    "epochs": 50,                   # Maximum training epochs
    "early_stopping_patience": 5,   # Stop if no improvement for 5 epochs
    "validation_split": 0.2,        # 20% data for validation
}

# GRU Marine Wave Prediction Model Configuration
MARINE_MODEL_CONFIG = {
    "sequence_length": 24,          # 24 hours of input data
    "forecast_horizon": 24,         # Predict next 24 hours
    "gru_units_layer1": 32,         # First GRU layer units
    "gru_units_layer2": 16,         # Second GRU layer units
    "dropout_rate": 0.2,            # Dropout for regularization
    "learning_rate": 0.001,         # Adam optimizer learning rate
    "batch_size": 32,               # Training batch size
    "epochs": 50,                   # Maximum training epochs
    "early_stopping_patience": 5,   # Stop if no improvement for 5 epochs
    "validation_split": 0.2,        # 20% data for validation
}

# =============================================================================
# SAFETY CLASSIFICATION THRESHOLDS
# =============================================================================
# Based on Indian Meteorological Department (IMD) guidelines and 
# international maritime safety standards

SAFETY_THRESHOLDS = {
    "safe": {
        "wave_height_max": 1.0,     # meters
        "wind_speed_max": 10.0,     # m/s (~20 knots)
        "color": "#28a745",         # Green
        "label": "Safe",
        "description": "Conditions favorable for fishing and coastal activities"
    },
    "caution": {
        "wave_height_min": 1.0,
        "wave_height_max": 2.5,     # meters
        "wind_speed_min": 10.0,
        "wind_speed_max": 17.0,     # m/s (~33 knots)
        "color": "#ffc107",         # Yellow/Amber
        "label": "Caution",
        "description": "Exercise caution, avoid venturing far from shore"
    },
    "dangerous": {
        "wave_height_min": 2.5,     # meters
        "wind_speed_min": 17.0,     # m/s
        "color": "#dc3545",         # Red
        "label": "Dangerous",
        "description": "Do not venture into sea, stay away from coast"
    }
}

# =============================================================================
# FILE PATHS
# =============================================================================

# Model save paths
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
CLIMATE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "climate_gru_model.keras")
MARINE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "marine_gru_model.keras")

# Scaler save paths for data normalization
SCALER_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scalers")
CLIMATE_SCALER_PATH = os.path.join(SCALER_SAVE_DIR, "climate_scaler.pkl")
MARINE_SCALER_PATH = os.path.join(SCALER_SAVE_DIR, "marine_scaler.pkl")

# Data cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

# =============================================================================
# FLASK API CONFIGURATION
# =============================================================================

FLASK_CONFIG = {
    "DEBUG": os.getenv("FLASK_DEBUG", "True").lower() == "true",
    "HOST": os.getenv("FLASK_HOST", "0.0.0.0"),
    "PORT": int(os.getenv("FLASK_PORT", 5000)),
    "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}
