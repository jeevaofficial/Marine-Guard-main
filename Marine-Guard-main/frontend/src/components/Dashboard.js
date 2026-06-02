/**
 * Main Dashboard Component
 * ========================
 * Integrates all sub-components for the marine safety dashboard
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React, { useState, useEffect, useCallback } from 'react';
import DistrictSelector from './DistrictSelector';
import CurrentConditions from './CurrentConditions';
import WaveForecastChart from './WaveForecastChart';
import SafetyIndicator from './SafetyIndicator';
import AIExplanation from './AIExplanation';
import DistrictOverview from './DistrictOverview';
import WindRose from './WindRose';
import { fetchData, getPrediction, getExplanation } from '../services/api';
import { useLanguage } from '../contexts/LanguageContext';
import './Dashboard.css';

const Dashboard = () => {
  const { language } = useLanguage(); // Get current language from context
  
  // State management
  const [selectedDistrict, setSelectedDistrict] = useState('Chennai');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Data states
  const [currentData, setCurrentData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [safetyData, setSafetyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [explanation, setExplanation] = useState(null);
  
  /**
   * Fetch all data for selected district
   */
  const loadDistrictData = useCallback(async (district) => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch current conditions and forecast
      const dataResponse = await fetchData(district);
      setCurrentData(dataResponse.current);
      setForecastData(dataResponse.forecast_24h);
      setSafetyData(dataResponse.safety);  // Store safety data from backend
      
      // Fetch predictions
      const predictionResponse = await getPrediction(district, 24);
      setPredictionData(predictionResponse);
      
      // Fetch AI explanation with current language
      const explanationResponse = await getExplanation(district, null, null, language);
      setExplanation(explanationResponse);
      
    } catch (err) {
      console.error('Error loading data:', err);
      setError(err.response?.data?.message || 'Failed to load data. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [language]);
  
  // Load data when district changes
  useEffect(() => {
    if (selectedDistrict) {
      loadDistrictData(selectedDistrict);
    }
  }, [selectedDistrict, loadDistrictData]);
  
  /**
   * Handle district selection change
   */
  const handleDistrictChange = (district) => {
    setSelectedDistrict(district);
  };
  
  /**
   * Handle refresh button click
   */
  const handleRefresh = () => {
    loadDistrictData(selectedDistrict);
  };
  
  return (
    <div className="dashboard">
      {/* Top Controls Section */}
      <div className="dashboard-controls">
        <div className="control-left">
          <DistrictSelector
            selectedDistrict={selectedDistrict}
            onDistrictChange={handleDistrictChange}
          />
        </div>
        <div className="control-right">
          <button 
            className="btn btn-primary refresh-btn"
            onClick={handleRefresh}
            disabled={loading}
          >
            {loading ? '⏳ Loading...' : '🔄 Refresh Data'}
          </button>
        </div>
      </div>
      
      {/* Error Display */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {/* Loading State */}
      {loading && !currentData && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p className="loading-text">Fetching marine data for {selectedDistrict}...</p>
        </div>
      )}
      
      {/* Main Dashboard Content */}
      {currentData && !loading && (
        <>
          {/* Safety Status - Prominent Display */}
          <div className="safety-section">
            <SafetyIndicator
              status={safetyData?.status}
              waveHeight={safetyData?.wave_height || forecastData?.wave_height?.max}
              windSpeed={safetyData?.wind_speed || forecastData?.wind_speed?.max}
            />
          </div>
          
          {/* Grid Layout for Cards */}
          <div className="dashboard-grid">
            {/* Current Conditions Card */}
            <div className="grid-item">
              <CurrentConditions
                district={selectedDistrict}
                data={currentData}
              />
            </div>
            
            {/* Marine Forecast Charts Container */}
            <div className="grid-item chart-item">
              <div className="marine-charts-container">
                <div className="charts-header">
                  <h3 className="charts-title">📊 Marine Forecast Analysis</h3>
                  <span className="forecast-badge">Next 24 Hours</span>
                </div>
                
                {/* Wave Height Forecast */}
                <div className="chart-section">
                  <WaveForecastChart
                    district={selectedDistrict}
                    predictions={predictionData?.predictions || forecastData?.hourly_wave_heights}
                    timestamps={predictionData?.timestamps || forecastData?.timestamps}
                    statistics={predictionData?.statistics || forecastData?.wave_height}
                  />
                </div>
                
                {/* Wind Rose */}
                <div className="chart-section">
                  <WindRose forecastData={forecastData} />
                </div>
              </div>
            </div>
            
            {/* AI Explanation Card */}
            <div className="grid-item explanation-item">
              <AIExplanation
                district={selectedDistrict}
                explanation={explanation?.explanation}
                safetyStatus={explanation?.safety_status}
                aiProvider={explanation?.ai_provider}
              />
            </div>
          </div>
          
          {/* All Districts Overview */}
          <div className="overview-section">
            <DistrictOverview
              currentDistrict={selectedDistrict}
              onDistrictSelect={handleDistrictChange}
            />
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
