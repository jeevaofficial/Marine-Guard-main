/**
 * Current Conditions Component
 * ============================
 * Displays real-time marine and weather conditions
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './CurrentConditions.css';

const CurrentConditions = ({ district, data }) => {
  const { t } = useLanguage();

  if (!data) {
    return (
      <div className="card conditions-card">
        <div className="card-header">
          <h3 className="card-title">📊 {t('currentConditions')}</h3>
        </div>
        <p className="no-data">{t('noExplanation')}</p>
      </div>
    );
  }

  // Format wind direction to cardinal
  const getWindDirection = (degrees) => {
    const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                       'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const index = Math.round((degrees % 360) / 22.5) % 16;
    return directions[index];
  };

  // Condition items with icons
  const conditions = [
    {
      icon: '🌊',
      label: t('waveHeight'),
      value: data.wave_height?.toFixed(2) || '-',
      unit: t('meters'),
      highlight: true,
    },
    {
      icon: '⏱️',
      label: t('wavePeriod'),
      value: data.wave_period?.toFixed(1) || '-',
      unit: t('seconds'),
    },
    {
      icon: '💨',
      label: t('windSpeed'),
      value: data.wind_speed?.toFixed(1) || '-',
      unit: 'm/s',
    },
    {
      icon: '🧭',
      label: t('windDirection'),
      value: data.wind_direction ? `${getWindDirection(data.wind_direction)} (${data.wind_direction}${t('degrees')})` : '-',
      unit: '',
    },
    {
      icon: '🌡️',
      label: t('temperature'),
      value: data.temperature?.toFixed(1) || '-',
      unit: t('celsius'),
    },
    {
      icon: '💧',
      label: t('humidity'),
      value: data.humidity?.toFixed(0) || '-',
      unit: t('percent'),
    },
    {
      icon: '📊',
      label: t('pressure'),
      value: data.pressure?.toFixed(1) || '-',
      unit: t('hPa'),
    },
  ];

  return (
    <div className="card conditions-card">
      <div className="card-header">
        <h3 className="card-title">📊 {t('currentConditions')}</h3>
        <span className="district-tag">{t(`districts.${district}`)}</span>
      </div>
      
      <div className="conditions-list">
        {conditions.map((item, index) => (
          <div 
            key={index} 
            className={`condition-item ${item.highlight ? 'highlight' : ''}`}
          >
            <span className="condition-icon">{item.icon}</span>
            <div className="condition-details">
              <span className="condition-label">{item.label}</span>
              <span className="condition-value">
                {item.value}
                <span className="condition-unit">{item.unit}</span>
              </span>
            </div>
          </div>
        ))}
      </div>
      
      <div className="conditions-footer">
        <span className="timestamp">
          {t('loading')}: {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
};

export default CurrentConditions;
