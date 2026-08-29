# AI-Based Marine & Coastal Safety Forecasting System for Tamil Nadu

## 🌊 Project Overview

This is a complete end-to-end AI-powered marine safety forecasting system designed for the Tamil Nadu coastal region in India. The system helps fishermen, coastal communities, and maritime authorities make informed decisions about sea conditions.

### Key Features

- **Real-time Marine Data**: Fetches current wave conditions from Open-Meteo Marine API
- **Historical Climate Data**: Uses NASA POWER API for training climate models
- **GRU Neural Networks**: Two lightweight GRU models for:
  - Climate forecasting (temperature, humidity, wind, pressure)
  - Wave height prediction (6-24 hours ahead)
- **Safety Classification**: Automatic classification into Safe/Caution/Dangerous
- **AI Explanations**: Groq AI GPT-4o integration for natural language explanations
- **Interactive Dashboard**: React-based frontend with charts and district overview

### Coverage

All 14 coastal districts of Tamil Nadu:
- Thiruvallur, Chennai, Kanchipuram, Chengalpattu
- Villupuram, Cuddalore, Mayiladuthurai, Nagapattinam
- Thanjavur, Tiruvarur, Ramanathapuram
- Thoothukudi, Tirunelveli, Kanniyakumari

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ District │ │ Forecast │ │  Safety  │ │  AI Explanation  │   │
│  │ Selector │ │  Chart   │ │ Indicator│ │      Panel       │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK BACKEND                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     API ENDPOINTS                         │  │
│  │  /fetch-data  │  /predict  │  /explain  │  /districts    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      SERVICES                             │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐   │  │
│  │  │ NASA POWER  │ │ Open-Meteo  │ │   Groq AI    │   │  │
│  │  │   Service   │ │   Service   │ │     Service       │   │  │
│  │  └─────────────┘ └─────────────┘ └───────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    GRU MODELS                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐              │  │
│  │  │ Climate Model   │    │  Marine Model   │              │  │
│  │  │ (32→16 GRU)     │    │  (32→16 GRU)    │              │  │
│  │  └─────────────────┘    └─────────────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  NASA POWER  │    │  Open-Meteo  │    │   Groq AI   │
│     API      │    │  Marine API  │    │     GPT-4o       │
│  (Historical)│    │  (Forecast)  │    │  (Explanations)  │
└──────────────┘    └──────────────┘    └──────────────────┘
```

---

## 📁 Project Structure

```
Marine Project/
├── backend/
│   ├── app.py                 # Flask application & API endpoints
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment variables template
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py        # All configuration settings
│   │   └── prompts.py         # GPT-4o prompt templates
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── climate_model.py   # GRU climate forecasting model
│   │   ├── marine_model.py    # GRU wave prediction model
│   │   ├── train_models.py    # Model training script
│   │   ├── saved/             # Saved model files (.keras)
│   │   └── scalers/           # Saved scaler files (.pkl)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nasa_power_service.py    # NASA POWER API client
│   │   ├── open_meteo_service.py    # Open-Meteo Marine API client
│   │   └── groq_service.py  # Groq AI GPT-4o client
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processor.py  # Data cleaning & preprocessing
│   │   └── helpers.py         # Utility functions
│   │
│   └── data/
│       └── cache/             # Cached API responses
│
├── frontend/
│   ├── package.json           # Node.js dependencies
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   │
│   └── src/
│       ├── App.js             # Main React component
│       ├── App.css            # Global styles
│       ├── index.js           # React entry point
│       │
│       ├── components/
│       │   ├── Dashboard.js           # Main dashboard
│       │   ├── DistrictSelector.js    # District dropdown
│       │   ├── CurrentConditions.js   # Current weather display
│       │   ├── WaveForecastChart.js   # Wave prediction chart
│       │   ├── SafetyIndicator.js     # Safety status banner
│       │   ├── AIExplanation.js       # GPT-4o explanation panel
│       │   └── DistrictOverview.js    # All districts summary
│       │
│       └── services/
│           └── api.js         # API client for Flask backend
│
└── README.md                  # This file
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.9+ (for backend)
- Node.js 18+ (for frontend)
- Git

### Step 1: Clone/Setup the Project

```bash
cd "d:\My Work\Project\Marine Project"
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
copy .env.example .env
# Edit .env and add your Groq AI credentials (optional)
```

### Step 3: Train Models (Optional but Recommended)

```bash
# Train models for Chennai district
python models/train_models.py --district Chennai --days 30

# Or train for all districts (takes longer)
python models/train_models.py --all --days 30
```

### Step 4: Start Backend Server

```bash
# Run Flask server
python app.py

# Server will start at http://localhost:5000
```

### Step 5: Frontend Setup (New Terminal)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start React development server
npm start

# Frontend will open at http://localhost:3000
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# Flask
FLASK_DEBUG=True
FLASK_PORT=5000

# Groq AI (optional - system works without it)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

```

### Safety Thresholds

The system classifies sea conditions based on wave height:

| Status | Wave Height | Color | Description |
|--------|-------------|-------|-------------|
| Safe | < 1.0 m | 🟢 Green | Favorable for fishing |
| Caution | 1.0 - 2.5 m | 🟡 Yellow | Exercise care |
| Dangerous | ≥ 2.5 m | 🔴 Red | Do not venture into sea |

---

## 📡 API Endpoints

### GET `/api/districts`
Returns list of all 14 coastal districts with coordinates.

### GET `/api/fetch-data/<district_name>`
Fetches current marine conditions for a specific district.

### POST `/api/predict`
Generates wave height predictions using the GRU model.
```json
{
  "district": "Chennai",
  "hours_ahead": 24
}
```

### POST `/api/explain`
Generates AI explanation for current conditions.
```json
{
  "district": "Chennai"
}
```

### GET `/api/all-districts`
Returns current conditions for all 14 districts.

---

## 🧠 Model Architecture

### Climate GRU Model
```
Input: (24 hours × 9 features)
    ↓
GRU Layer 1: 32 units, return_sequences=True
    ↓
Dropout: 20%
    ↓
GRU Layer 2: 16 units
    ↓
Dropout: 20%
    ↓
Dense Output: 12 values (forecast horizon)
```

### Marine GRU Model
```
Input: (24 hours × features)
    ↓
GRU Layer 1: 32 units, return_sequences=True
    ↓
Dropout: 20%
    ↓
GRU Layer 2: 16 units
    ↓
Dropout: 20%
    ↓
Dense Hidden: 32 units, ReLU
    ↓
Dense Output: 24 values (hourly wave heights)
```

### Why GRU over LSTM?
1. **Faster training** - Fewer parameters (2 gates vs 3)
2. **Similar performance** - For sequences < 100 timesteps
3. **Lower memory** - Better for CPU training
4. **Simpler architecture** - Easier to explain in viva

---

## 📊 Data Sources

### NASA POWER API (Historical Climate)
- **URL**: https://power.larc.nasa.gov/api/
- **Parameters**: T2M, RH2M, WS2M, PS, PRECTOTCORR
- **Resolution**: Hourly
- **Cost**: FREE, no API key required
- **Use**: Training climate forecasting model

### Open-Meteo Marine API (Forecasts)
- **URL**: https://marine-api.open-meteo.com/v1/marine
- **Parameters**: wave_height, wave_period, wave_direction
- **Resolution**: Hourly, 7-day forecast
- **Cost**: FREE, no API key required
- **Use**: Current conditions + future marine data

### Groq AI GPT-4o (Explanations)
- **Model**: GPT-4o
- **Use**: Natural language safety explanations
- **Cost**: Pay-per-use (optional)
- **Fallback**: Rule-based explanations if not configured

---

## 🎯 Key Features for Viva

### 1. Data Pipeline
- NASA POWER provides **historical** data only
- Climate variables **must be forecasted** using GRU
- Open-Meteo provides **real-time + forecast** marine data
- Both datasets **aligned by timestamp** before feeding to model

### 2. Model Design Choices
- **GRU chosen over LSTM** for faster CPU training
- **32→16 unit architecture** balances accuracy and speed
- **Early stopping** prevents overfitting
- **MinMax scaling** ensures proper gradient flow

### 3. Safety Classification
- Based on **IMD guidelines** and maritime standards
- **Wave height** is primary factor
- **Wind speed** can upgrade danger level
- **Color-coded** for quick understanding

### 4. AI Integration
- **GPT-4o** provides human-readable explanations
- **Fallback system** works without Groq credentials
- **Prompt engineering** ensures consistent, helpful responses

---

## ⚠️ Assumptions & Limitations

### Assumptions
1. Wave height is the primary safety indicator
2. Historical patterns from past 30-60 days are representative
3. Open-Meteo forecast accuracy is acceptable for this use case
4. Coastal district coordinates represent the main fishing areas

### Limitations
1. **Model accuracy**: GRU predictions may have ±0.3m error
2. **Data lag**: NASA POWER has ~5 day data lag
3. **Coverage**: Coastal coordinates are approximate
4. **Real-time**: Not suitable for emergency decisions without verification
5. **Training data**: Limited to available historical data

### Important Disclaimer
This system is for **educational and informational purposes**. For actual maritime decisions, always consult:
- Indian Meteorological Department (IMD)
- Indian National Centre for Ocean Information Services (INCOIS)
- Local fishing authorities

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend API errors
```bash
# Ensure backend is running on port 5000
# Check CORS is enabled in Flask
# Verify proxy setting in package.json
```

### Model not found
```bash
# Train model first
python models/train_models.py --district Chennai
```

### Groq AI not working
- System will use fallback explanations
- Check .env file for correct credentials
- Verify deployment name matches your Groq setup

---

## 📝 License

This project is developed for B.Tech AI&DS .

---

## 👨‍💻 Author

B.Tech AI&DS 

---

## 🙏 Acknowledgments

- NASA POWER Project for historical climate data
- Open-Meteo for free marine forecast API
- TensorFlow/Keras team for deep learning framework
- Indian Meteorological Department for safety guidelines
