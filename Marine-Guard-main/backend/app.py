"""
Flask REST API for Marine Safety Forecasting System
====================================================
This module implements the main Flask application with REST API endpoints.

Endpoints:
- GET /api/districts - List all coastal districts
- GET /api/fetch-data/<district> - Fetch current conditions
- POST /api/predict - Generate wave predictions
- POST /api/explain - Get GPT-4o explanation
- GET /api/health - Health check endpoint

Author: B.Tech AI&DS Project
Date: 2026
"""

import os
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import FLASK_CONFIG, COASTAL_DISTRICTS, MODEL_SAVE_DIR, SCALER_SAVE_DIR
from services.open_meteo_service import OpenMeteoMarineService
from services.nasa_power_service import NASAPowerService
from services.azure_openai_service import AzureOpenAIService
from models.marine_model import MarineGRUModel
from utils.helpers import classify_safety_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, origins=FLASK_CONFIG.get("CORS_ORIGINS", ["http://localhost:3000"]))

# Initialize services
marine_service = OpenMeteoMarineService()
nasa_service = NASAPowerService()
openai_service = AzureOpenAIService()

# Cache for loaded models
loaded_models = {}


def get_marine_model(district_name: str) -> MarineGRUModel:
    """
    Get or load marine model for a district.
    
    Uses caching to avoid reloading models on every request.
    Falls back to a default model if district-specific model unavailable.
    """
    cache_key = f"marine_{district_name.lower()}"
    
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    model = MarineGRUModel()
    
    # Try district-specific model first
    model_path = os.path.join(MODEL_SAVE_DIR, f"marine_gru_{district_name.lower()}.keras")
    scaler_path = os.path.join(SCALER_SAVE_DIR, f"marine_scaler_{district_name.lower()}.pkl")
    
    # Fall back to default Chennai model
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_SAVE_DIR, "marine_gru_chennai.keras")
        scaler_path = os.path.join(SCALER_SAVE_DIR, "marine_scaler_chennai.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model.load(model_path, scaler_path)
            loaded_models[cache_key] = model
            logger.info(f"Loaded model for {district_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    else:
        logger.warning(f"No model found for {district_name}")
        return None
    
    return model


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON with system status
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "marine_api": "available",
            "nasa_api": "available",
            "openai": "configured" if openai_service.is_available() else "not_configured"
        },
        "version": "1.0.0"
    })


@app.route('/api/districts', methods=['GET'])
def list_districts():
    """
    Get list of all 14 coastal districts.
    
    Returns:
        JSON array of district information
    """
    districts = []
    for name, info in COASTAL_DISTRICTS.items():
        districts.append({
            "name": name,
            "latitude": info["lat"],
            "longitude": info["lon"],
            "description": info["description"]
        })
    
    return jsonify({
        "count": len(districts),
        "districts": districts
    })


@app.route('/api/fetch-data/<district_name>', methods=['GET'])
def fetch_data(district_name: str):
    """
    Fetch current marine and weather conditions for a district.
    
    Args:
        district_name: Name of the coastal district
        
    Returns:
        JSON with current conditions and short-term forecast
    """
    # Validate district
    if district_name not in COASTAL_DISTRICTS:
        return jsonify({
            "error": "Invalid district",
            "message": f"District '{district_name}' not found",
            "valid_districts": list(COASTAL_DISTRICTS.keys())
        }), 400
    
    try:
        # Get current conditions
        current = marine_service.get_current_conditions(district_name)
        
        # Get forecast summary
        forecast = marine_service.get_forecast_summary(district_name, hours_ahead=24)
        
        # Classify safety
        max_wave = forecast["wave_height"]["max"]
        wind_speed = forecast.get("wind_speed", {}).get("max")
        safety = classify_safety_status(max_wave, wind_speed)
        
        return jsonify({
            "district": district_name,
            "timestamp": datetime.now().isoformat(),
            "current": current,
            "forecast_24h": forecast,
            "safety": safety
        })
        
    except Exception as e:
        logger.error(f"Error fetching data for {district_name}: {e}")
        return jsonify({
            "error": "Data fetch failed",
            "message": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate wave height predictions using GRU model.
    
    Request body:
        {
            "district": "Chennai",
            "hours_ahead": 24  // optional, default 24
        }
        
    Returns:
        JSON with hourly predictions and safety classification
    """
    data = request.get_json()
    
    if not data or 'district' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide 'district' in request body"
        }), 400
    
    district_name = data['district']
    hours_ahead = data.get('hours_ahead', 24)
    
    # Validate district
    if district_name not in COASTAL_DISTRICTS:
        return jsonify({
            "error": "Invalid district",
            "message": f"District '{district_name}' not found"
        }), 400
    
    try:
        # Get recent data for prediction
        recent_df = marine_service.fetch_for_district(district_name, forecast_days=2)
        
        # Get or load model
        model = get_marine_model(district_name)
        
        if model is None:
            # Fallback to API forecast if model not available
            logger.warning(f"Model not available, using API forecast")
            forecast = marine_service.get_forecast_summary(district_name, hours_ahead)
            
            return jsonify({
                "district": district_name,
                "method": "api_forecast",
                "predictions": forecast["hourly_wave_heights"][:hours_ahead],
                "timestamps": forecast["timestamps"][:hours_ahead],
                "statistics": forecast["wave_height"],
                "safety": classify_safety_status(forecast["wave_height"]["max"])
            })
        
        # Make prediction with model
        result = model.predict_with_safety(recent_df)
        
        # Limit to requested hours
        result["predictions"] = result["predictions"][:hours_ahead]
        result["timestamps"] = result["timestamps"][:hours_ahead]
        
        result["district"] = district_name
        result["method"] = "gru_model"
        result["model_info"] = {
            "type": "GRU",
            "layers": 2,
            "units": [32, 16]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error for {district_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Generate GPT-4o explanation of marine conditions.
    
    Request body:
        {
            "district": "Chennai",
            "current_data": {...},  // optional
            "forecast_data": {...}, // optional
            "language": "en"        // optional, 'en' or 'ta', default 'en'
        }
        
    Returns:
        JSON with AI-generated explanation and recommendations
    """
    data = request.get_json()
    
    if not data or 'district' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide 'district' in request body"
        }), 400
    
    district_name = data['district']
    language = data.get('language', 'en')  # Get language from request, default to English
    
    # Validate district
    if district_name not in COASTAL_DISTRICTS:
        return jsonify({
            "error": "Invalid district"
        }), 400
    
    try:
        # Get current data if not provided
        if 'current_data' not in data:
            current = marine_service.get_current_conditions(district_name)
        else:
            current = data['current_data']
        
        # Get forecast data if not provided
        if 'forecast_data' not in data:
            forecast = marine_service.get_forecast_summary(district_name, hours_ahead=24)
            forecast_data = {
                "max_wave_height": forecast["wave_height"]["max"],
                "avg_wave_height": forecast["wave_height"]["mean"],
                "max_wind_speed": forecast.get("wind_speed", {}).get("max", 10)
            }
        else:
            forecast_data = data['forecast_data']
        
        # Classify safety
        max_wave = forecast_data.get("max_wave_height", current.get("wave_height", 1))
        wind_speed = current.get("wind_speed")
        safety = classify_safety_status(max_wave, wind_speed)
        
        # Generate explanation with language parameter
        explanation = openai_service.generate_explanation(
            district=district_name,
            current_data=current,
            forecast_data=forecast_data,
            safety_status=safety["status"],
            language=language
        )
        
        return jsonify({
            "district": district_name,
            "timestamp": datetime.now().isoformat(),
            "safety_status": safety["status"],
            "safety_color": safety["color"],
            "explanation": explanation,
            "language": language,
            "ai_provider": "azure_openai" if openai_service.is_available() else "fallback",
            "current_conditions": current,
            "forecast_summary": forecast_data
        })
        
    except Exception as e:
        logger.error(f"Explanation error for {district_name}: {e}")
        return jsonify({
            "error": "Explanation generation failed",
            "message": str(e)
        }), 500


@app.route('/api/all-districts', methods=['GET'])
def all_districts_status():
    """
    Get current conditions for all 14 coastal districts.
    Uses parallel fetching for better performance.
    
    Returns:
        JSON with conditions for all districts
    """
    def fetch_district_status(district_name):
        """Fetch status for a single district."""
        try:
            current = marine_service.get_current_conditions(district_name)
            
            # Try to get forecast, but don't fail if it times out
            try:
                forecast = marine_service.get_forecast_summary(district_name, hours_ahead=24)
                max_wave = forecast.get("wave_height", {}).get("max", current.get("wave_height", 0))
                max_wind = forecast.get("wind_speed", {}).get("max", current.get("wind_speed"))
            except Exception as forecast_err:
                logger.warning(f"Forecast fetch failed for {district_name}, using current: {forecast_err}")
                max_wave = current.get("wave_height", 0)
                max_wind = current.get("wind_speed")
            
            # Use max forecast values for safety classification (same as main dashboard)
            safety = classify_safety_status(max_wave, max_wind)
            
            return {
                "district": district_name,
                "wave_height": current.get("wave_height"),
                "wind_speed": current.get("wind_speed"),
                "max_wave_height": max_wave,
                "safety_status": safety["status"],
                "safety_color": safety["color"],
                "data_unavailable": current.get("data_unavailable", False)
            }
        except Exception as e:
            logger.error(f"Error for {district_name}: {e}")
            return {
                "district": district_name,
                "wave_height": 0,
                "wind_speed": 0,
                "safety_status": "Unknown",
                "safety_color": "#gray",
                "error": str(e)
            }
    
    try:
        results = []
        
        # Use ThreadPoolExecutor for parallel fetching (max 7 concurrent for better throughput)
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {
                executor.submit(fetch_district_status, name): name 
                for name in COASTAL_DISTRICTS.keys()
            }
            
            for future in as_completed(futures, timeout=90):
                try:
                    result = future.result(timeout=20)
                    results.append(result)
                except Exception as e:
                    district_name = futures[future]
                    logger.error(f"Timeout for {district_name}: {e}")
                    results.append({
                        "district": district_name,
                        "wave_height": 0,
                        "wind_speed": 0,
                        "safety_status": "Unknown",
                        "safety_color": "gray",
                        "error": "Timeout"
                    })
        
        # Sort by district name for consistent ordering
        results.sort(key=lambda x: x["district"])
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "count": len(results),
            "districts": results
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to fetch all districts",
            "message": str(e)
        }), 500


@app.route('/api/historical/<district_name>', methods=['GET'])
def historical_data(district_name: str):
    """
    Get historical climate data for a district.
    
    Query params:
        days: Number of days of historical data (default: 7)
        
    Returns:
        JSON with historical climate data
    """
    if district_name not in COASTAL_DISTRICTS:
        return jsonify({"error": "Invalid district"}), 400
    
    days = request.args.get('days', 7, type=int)
    days = min(days, 30)  # Limit to 30 days
    
    try:
        df = nasa_service.fetch_for_district(district_name, days_back=days)
        
        return jsonify({
            "district": district_name,
            "days": days,
            "record_count": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "data": {
                "timestamps": [t.isoformat() for t in df.index],
                "temperature": df['T2M'].tolist() if 'T2M' in df.columns else [],
                "humidity": df['RH2M'].tolist() if 'RH2M' in df.columns else [],
                "wind_speed": df['WS2M'].tolist() if 'WS2M' in df.columns else [],
                "pressure": df['PS'].tolist() if 'PS' in df.columns else [],
                "precipitation": df['PRECTOTCORR'].tolist() if 'PRECTOTCORR' in df.columns else []
            }
        })
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({
            "error": "Failed to fetch historical data",
            "message": str(e)
        }), 500


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """
    Trigger model training for a district.
    
    Request body:
        {
            "district": "Chennai",
            "days_back": 30  // optional
        }
        
    Note: This is a long-running operation.
    """
    data = request.get_json()
    
    if not data or 'district' not in data:
        return jsonify({"error": "Missing district"}), 400
    
    district_name = data['district']
    days_back = data.get('days_back', 30)
    
    if district_name not in COASTAL_DISTRICTS:
        return jsonify({"error": "Invalid district"}), 400
    
    try:
        from models.train_models import train_single_district
        
        # This is a synchronous call - in production, use a task queue
        results = train_single_district(district_name, days_back)
        
        return jsonify({
            "status": "completed",
            "district": district_name,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({
            "error": "Training failed",
            "message": str(e)
        }), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MARINE SAFETY FORECASTING API")
    print("="*60)
    print(f"\nStarting Flask server...")
    print(f"Host: {FLASK_CONFIG.get('HOST', '0.0.0.0')}")
    print(f"Port: {FLASK_CONFIG.get('PORT', 5000)}")
    print(f"Debug: {FLASK_CONFIG.get('DEBUG', True)}")
    print(f"\nAvailable districts: {len(COASTAL_DISTRICTS)}")
    print(f"Azure OpenAI: {'Configured' if openai_service.is_available() else 'Not configured'}")
    print("\n" + "="*60)
    
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False

    )
