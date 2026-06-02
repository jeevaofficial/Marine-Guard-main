/**
 * District Overview Component
 * ===========================
 * Shows safety status overview for all 14 coastal districts
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { getAllDistrictsStatus } from '../services/api';
import './DistrictOverview.css';

const DistrictOverview = ({ currentDistrict, onDistrictSelect }) => {
  const { t } = useLanguage();
  const [districts, setDistricts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    loadAllDistricts();
  }, []);

  const loadAllDistricts = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getAllDistrictsStatus();
      setDistricts(response.districts || []);
      setError(null);
      setRetryCount(0);
    } catch (err) {
      console.error('Error loading districts:', err);
      if (retryCount < 2) {
        // Auto-retry up to 2 times
        setRetryCount(prev => prev + 1);
        setTimeout(() => loadAllDistricts(), 2000);
        return;
      }
      setError(t('clickToRetry'));
    } finally {
      setLoading(false);
    }
  };

  const getStatusClass = (status) => {
    switch (status?.toLowerCase()) {
      case 'safe':
        return 'safe';
      case 'caution':
        return 'caution';
      case 'dangerous':
        return 'dangerous';
      default:
        return 'unknown';
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'safe':
        return '✅';
      case 'caution':
        return '⚠️';
      case 'dangerous':
        return '🚨';
      default:
        return '❓';
    }
  };

  return (
    <div className="card overview-card">
      <div className="card-header">
        <h3 className="card-title">🗺️ {t('districtOverview')}</h3>
        <button 
          className="btn btn-small"
          onClick={loadAllDistricts}
          disabled={loading}
        >
          {loading ? '⏳' : '🔄'} {t('refresh')}
        </button>
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {loading && districts.length === 0 ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p className="loading-text">{t('loadingDistricts')}</p>
        </div>
      ) : (
        <div className="districts-grid">
          {districts.map((district) => (
            <div
              key={district.district}
              className={`district-item ${getStatusClass(district.safety_status)} ${
                district.district === currentDistrict ? 'selected' : ''
              }`}
              onClick={() => onDistrictSelect(district.district)}
            >
              <div className="district-header">
                <span className="district-icon">{getStatusIcon(district.safety_status)}</span>
                <span className="district-name">{district.district}</span>
              </div>
              <div className="district-stats">
                {district.error ? (
                  <span className="district-error">Error loading</span>
                ) : (
                  <>
                    <span className="stat">
                      🌊 {district.wave_height?.toFixed(1) || '-'}m
                    </span>
                    {district.wind_speed && (
                      <span className="stat">
                        💨 {district.wind_speed?.toFixed(0)}m/s
                      </span>
                    )}
                  </>
                )}
              </div>
              <div className="district-status">
                <span 
                  className="status-badge"
                  style={{ backgroundColor: district.safety_color || '#6c757d' }}
                >
                  {district.safety_status || 'Unknown'}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="overview-legend">
        <span className="legend-item">
          <span className="legend-dot safe"></span> Safe
        </span>
        <span className="legend-item">
          <span className="legend-dot caution"></span> Caution
        </span>
        <span className="legend-item">
          <span className="legend-dot dangerous"></span> Dangerous
        </span>
        <span className="legend-note">Click on a district to view details</span>
      </div>
    </div>
  );
};

export default DistrictOverview;
