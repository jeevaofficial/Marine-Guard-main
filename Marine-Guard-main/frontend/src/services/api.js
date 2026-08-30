/**
 * API Service Module
 * ==================
 * Handles all HTTP requests to the Flask backend
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import axios from 'axios';

const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'production'
    ? 'https://marine-guard-main.onrender.com/api'
    : '/api');

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 90000, // 90 second timeout for slow endpoints
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * API Methods
 */

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Get list of all districts
export const getDistricts = async () => {
  const response = await api.get('/districts');
  return response.data;
};

// Fetch current conditions for a district
export const fetchData = async (districtName) => {
  const response = await api.get(`/fetch-data/${districtName}`);
  return response.data;
};

// Get wave predictions for a district
export const getPrediction = async (districtName, hoursAhead = 24) => {
  const response = await api.post('/predict', {
    district: districtName,
    hours_ahead: hoursAhead,
  });
  return response.data;
};

// Get AI explanation for conditions
export const getExplanation = async (districtName, currentData = null, forecastData = null, language = 'en') => {
  const payload = {
    district: districtName,
    language: language,  // Add language parameter
  };
  
  if (currentData) {
    payload.current_data = currentData;
  }
  
  if (forecastData) {
    payload.forecast_data = forecastData;
  }
  
  const response = await api.post('/explain', payload);
  return response.data;
};

// Get all districts status
export const getAllDistrictsStatus = async () => {
  const response = await api.get('/all-districts');
  return response.data;
};

// Get historical data for a district
export const getHistoricalData = async (districtName, days = 7) => {
  const response = await api.get(`/historical/${districtName}?days=${days}`);
  return response.data;
};

export default api;
