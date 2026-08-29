# AI-Based Marine and Coastal Safety Forecasting System for Tamil Nadu

## 1. Project Summary
This project is an end-to-end marine safety intelligence platform for 14 coastal districts of Tamil Nadu. It combines real-time marine data, short-term forecasting, machine learning prediction, and AI-generated safety explanations to support safer fishing and coastal operations.

Core outcome:
- Provide district-level safety assessment as Safe, Caution, or Dangerous.
- Visualize current marine conditions and next-24-hour forecast.
- Explain safety status in English or Tamil.
- Offer a multi-district overview for rapid monitoring.

## 2. Problem Statement
Traditional fishing communities and local coastal operators often depend on fragmented weather signals, delayed advisories, or manual interpretation. This project addresses the gap by unifying:
- Live marine indicators
- Forecast trends
- ML-driven wave prediction
- Human-readable safety recommendations

## 3. Objectives
- Build a practical forecasting dashboard for Tamil Nadu coastal districts.
- Predict wave height for the next 24 hours using GRU models.
- Classify safety risk with explainable thresholds.
- Provide bilingual outputs (English and Tamil).
- Keep the system operational even when optional AI services are unavailable.

## 4. Coverage and Scope
Districts covered:
- Thiruvallur
- Chennai
- Kanchipuram
- Chengalpattu
- Villupuram
- Cuddalore
- Mayiladuthurai
- Nagapattinam
- Thanjavur
- Tiruvarur
- Ramanathapuram
- Thoothukudi
- Tirunelveli
- Kanniyakumari

Time horizon:
- Live conditions (near-current)
- Forecast window: typically 24 hours

## 5. Technology Stack
Backend:
- Flask 3
- Flask-CORS
- TensorFlow/Keras (GRU models)
- NumPy, Pandas, scikit-learn
- OpenAI SDK (Groq AI integration)

Frontend:
- React 18
- Axios
- Recharts

External data and AI services:
- Open-Meteo Marine API (marine and weather forecast)
- NASA POWER API (historical climate data)
- Groq AI GPT-4o (optional explanation generation)

## 6. High-Level Architecture
1. User selects district in React UI.
2. Frontend calls backend APIs for current data, forecast/prediction, and explanation.
3. Flask orchestrates calls to Open-Meteo service and model inference.
4. Safety logic classifies risk from forecast maxima.
5. Frontend renders cards/charts and district-level summaries.

Main modules:
- Data services layer
- Model layer (climate GRU and marine GRU)
- REST API layer
- React presentation layer with i18n context

## 7. Repository Structure
- backend/: Flask API, services, models, utilities
- frontend/: React application, components, i18n, API client
- README.md: project-level documentation

Important implementation files:
- backend/app.py
- backend/config/settings.py
- backend/services/open_meteo_service.py
- backend/services/nasa_power_service.py
- backend/services/groq_service.py
- backend/models/marine_model.py
- backend/models/climate_model.py
- backend/models/train_models.py
- frontend/src/components/Dashboard.js
- frontend/src/components/SafetyIndicator.js
- frontend/src/components/WaveForecastChart.js
- frontend/src/components/WindRose.js
- frontend/src/contexts/LanguageContext.js
- frontend/src/i18n/translations.js

## 8. Backend Design
### 8.1 API Endpoints
Health and metadata:
- GET /api/health
- GET /api/districts

Forecast and intelligence:
- GET /api/fetch-data/<district_name>
- POST /api/predict
- POST /api/explain
- GET /api/all-districts
- GET /api/historical/<district_name>
- POST /api/train

### 8.2 Endpoint Behavior
GET /api/fetch-data/<district_name>:
- Fetches current conditions and forecast summary.
- Computes safety from forecast maximums.
- Returns district, timestamp, current, forecast_24h, safety.

POST /api/predict:
- Uses district GRU model if available.
- Falls back to API forecast when model files are unavailable.
- Returns hourly predictions, timestamps, summary statistics, safety.

POST /api/explain:
- Uses Groq AI if configured.
- Falls back to deterministic rule-based explanation otherwise.
- Supports language parameter en or ta.

GET /api/all-districts:
- Parallel district fetch with ThreadPoolExecutor.
- Includes timeout handling and fallback safety values.

GET /api/historical/<district_name>:
- Returns NASA POWER historical dataset (bounded by days parameter).

POST /api/train:
- Triggers district model training from backend side.

## 9. Service Layer
### 9.1 OpenMeteoMarineService
Responsibilities:
- Fetch marine forecast and weather forecast.
- Merge into combined hourly dataset.
- Provide current condition snapshot.
- Produce 24-hour forecast summary and hourly data points.

Highlights:
- Handles missing/empty responses.
- Includes hourly wind direction and wind speed for wind rose.
- Uses nearest timestamp to current time.

### 9.2 NASAPowerService
Responsibilities:
- Fetch historical climate data with retry logic.
- Parse NASA POWER JSON payload.
- Prepare district-specific training data.

Highlights:
- Accounts for NASA data lag (about 5 days).
- Adds time features and derived features through utilities.

### 9.3 GroqService
Responsibilities:
- Generate natural-language safety explanation.
- Generate emergency warning content.
- Graceful fallback if Groq credentials are missing.

Highlights:
- Bilingual system prompt support.
- Controlled generation settings for safety-focused responses.

## 10. Machine Learning Layer
### 10.1 ClimateGRUModel
Purpose:
- Forecast climate variables from historical climate sequence.

Architecture:
- GRU(32) -> Dropout -> GRU(16) -> Dropout -> Dense output

Pipeline:
- Feature selection and normalization
- Sequence construction
- Train/validation split
- Early stopping and LR scheduling

### 10.2 MarineGRUModel
Purpose:
- Forecast wave heights for next 24 hours.

Architecture:
- GRU(32) -> Dropout -> GRU(16) -> Dropout -> Dense(32) -> Dense(horizon)

Pipeline:
- Merge marine and optional climate features
- Add cyclical time features
- Normalize and sequence
- Predict and inverse-scale
- Safety classification on peak forecast wave

### 10.3 Training Script
- backend/models/train_models.py supports:
  - Single district training
  - All district training
- Generates training report JSON
- Stores models and scalers in backend/models/saved and backend/models/scalers

## 11. Safety Classification Logic
Primary thresholds:
- Safe: wave < 1.0 m
- Caution: 1.0 m <= wave < 2.5 m
- Dangerous: wave >= 2.5 m

Wind escalation:
- If wind speed > 17 m/s and current class below Dangerous, upgrade to Dangerous.

Output includes:
- status
- color
- description
- risk_level
- rounded wave and wind values

## 12. Frontend Design
### 12.1 Core Components
- Dashboard: orchestration and state management
- DistrictSelector: district input control
- CurrentConditions: current marine/weather card
- SafetyIndicator: prominent risk panel
- WaveForecastChart: line and area forecast chart
- WindRose: directional wind distribution
- AIExplanation: textual AI/rule-based narrative
- DistrictOverview: all-district cards with status

### 12.2 State and Data Flow
Dashboard fetches:
- fetchData for current and forecast summary
- getPrediction for model/API predictions
- getExplanation for AI narrative

Display strategy:
- SafetyIndicator uses backend safety object for consistency.
- Chart uses prediction output when present, else forecast hourly wave values.

### 12.3 i18n
- Language context stores selected language in localStorage.
- Translations contain English and Tamil labels/phrases.
- UI toggle switches en <-> ta globally.

## 13. API Contracts (Practical)
POST /api/predict request:
- district: string
- hours_ahead: integer (default 24)

POST /api/explain request:
- district: string
- current_data: object (optional)
- forecast_data: object (optional)
- language: en or ta (optional)

GET /api/all-districts response (per district):
- district
- wave_height
- wind_speed
- max_wave_height
- safety_status
- safety_color
- optional error field

## 14. Setup and Run Guide
### Backend
1. cd backend
2. python -m venv venv
3. Activate venv
4. pip install -r requirements.txt
5. Configure .env for Groq AI (optional)
6. python app.py

### Frontend
1. cd frontend
2. npm install
3. npm start

Default URLs:
- Backend: http://localhost:5000
- Frontend: http://localhost:3000

## 15. Configuration
Key environment variables:
- FLASK_DEBUG
- FLASK_HOST
- FLASK_PORT
- CORS_ORIGINS
- GROQ_ENDPOINT
- GROQ_API_KEY
- GROQ_DEPLOYMENT
- GROQ_API_VERSION

## 16. Performance and Reliability Notes
- /api/all-districts uses concurrency to reduce total fetch time.
- Forecast retrieval has fallback when per-district forecast fails.
- AI explanation has rule-based fallback, preserving availability.
- Model loading is cached in-process to avoid repeated disk load.

## 17. Security and Operational Considerations
- Keep Groq API keys only in environment variables.
- Restrict CORS origins in production.
- Add API rate limits for public deployment.
- Add structured request logging and monitoring for production.

## 18. Current Strengths
- Clear modular separation (services, models, API, UI).
- Practical fallback behavior when optional components fail.
- Multi-language support integrated at context level.
- Strong visual analytics for decision support (wave chart + wind rose + district overview).

## 19. Current Gaps and Improvement Opportunities
- No persistent prediction-vs-actual storage for model performance tracking over time.
- No automated unit/integration test suite committed.
- Some UI labels remain hardcoded in a few components and can be fully centralized in translations.
- App header language hook usage pattern should be standardized to direct import style for maintainability.
- Forecast and explanation requests are sequential in Dashboard and can be optimized for lower latency.

## 20. Suggested Next Milestones
1. Add database table or CSV logging for prediction and later observed values.
2. Build evaluation pipeline with MAE/RMSE dashboard by district.
3. Add backend tests for safety classifier and API contracts.
4. Add frontend tests for language toggle and critical rendering paths.
5. Add alerting channel (SMS/WhatsApp/push) for dangerous conditions.

## 21. VIVA Talking Points
- Why GRU: lower parameter count, faster CPU training, suitable for hourly sequence forecasting.
- Why two data sources: historical climate context + live marine operational conditions.
- Why fallback design: system remains useful even without cloud AI availability.
- Why threshold-based safety: explainable and policy-aligned decisions for field users.
- Why bilingual UI: direct usability for Tamil fishing communities.

## 22. Conclusion
This project delivers a practical and extensible marine safety intelligence platform with real-time monitoring, forecasting, explainable risk classification, and multilingual communication. It is suitable as a deployable base for coastal advisory systems and can be strengthened further through persistent evaluation logging, test automation, and production hardening.
