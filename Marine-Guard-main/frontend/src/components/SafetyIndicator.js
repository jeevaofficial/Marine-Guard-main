/**
 * Safety Indicator Component
 * ==========================
 * Prominent display of current safety status
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './SafetyIndicator.css';

const SafetyIndicator = ({ status, waveHeight, windSpeed }) => {
  const { t } = useLanguage();
  
  // Determine status details
  const getStatusDetails = () => {
    switch (status?.toLowerCase()) {
      case 'safe':
        return {
          icon: '✅',
          title: t('safe').toUpperCase(),
          subtitle: t('safeDesc'),
          className: 'safe',
          recommendations: t('safeRecommendations'),
        };
      case 'caution':
        return {
          icon: '⚠️',
          title: t('caution').toUpperCase(),
          subtitle: t('cautionDesc'),
          className: 'caution',
          recommendations: t('cautionRecommendations'),
        };
      case 'dangerous':
        return {
          icon: '🚨',
          title: t('dangerous').toUpperCase(),
          subtitle: t('dangerousDesc'),
          className: 'dangerous',
          recommendations: t('dangerousRecommendations'),
        };
      default:
        return {
          icon: '❓',
          title: 'UNKNOWN',
          subtitle: t('unknownStatus'),
          className: 'unknown',
          recommendations: t('unknownRecommendations'),
        };
    }
  };

  const details = getStatusDetails();

  return (
    <div className={`safety-indicator ${details.className}`}>
      <div className="safety-main">
        <div className="safety-status-section">
          <span className="safety-icon">{details.icon}</span>
          <div className="safety-text">
            <h2 className="safety-title">{details.title}</h2>
            <p className="safety-subtitle">{details.subtitle}</p>
          </div>
        </div>
        
        <div className="safety-metrics">
          <div className="metric">
            <span className="metric-label">{t('waveHeight')}</span>
            <span className="metric-value">{waveHeight?.toFixed(2) || '-'} {t('meters')}</span>
          </div>
          {windSpeed && (
            <div className="metric">
              <span className="metric-label">{t('windSpeed')}</span>
              <span className="metric-value">{windSpeed?.toFixed(1)} {t('ms')}</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="safety-recommendations">
        <h4>{t('recommendationsTitle')}</h4>
        <ul>
          {details.recommendations.map((rec, index) => (
            <li key={index}>{rec}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default SafetyIndicator;
