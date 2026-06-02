/**
 * Wave Forecast Chart Component
 * =============================
 * Displays wave height predictions in an interactive chart
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { translations } from '../i18n/translations';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import './WaveForecastChart.css';

const WaveForecastChart = ({ district, predictions, timestamps, statistics }) => {
  const { language } = useLanguage();
  
  // Prepare chart data
  const chartData = predictions?.map((value, index) => {
    const time = timestamps?.[index] 
      ? new Date(timestamps[index])
      : new Date(Date.now() + (index + 1) * 3600000);
    
    return {
      hour: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      fullTime: time.toLocaleString(),
      waveHeight: parseFloat(value?.toFixed(2) || 0),
      // Safety zones
      safeZone: 1.0,
      cautionZone: 2.5,
    };
  }) || [];

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const waveHeight = data.waveHeight;
      
      let status = translations[language].safe;
      let statusColor = '#28a745';
      if (waveHeight >= 2.5) {
        status = translations[language].dangerous;
        statusColor = '#dc3545';
      } else if (waveHeight >= 1.0) {
        status = translations[language].caution;
        statusColor = '#ffc107';
      }
      
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{data.fullTime}</p>
          <p className="tooltip-value">
            {translations[language].waveHeightLabel}: <strong>{waveHeight} {translations[language].meters}</strong>
          </p>
          <p className="tooltip-status" style={{ color: statusColor }}>
            {translations[language].statusLabel}: {status}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-card-content">
      <div className="chart-subheader">
        <h4 className="chart-subtitle">📈 {translations[language].waveForecastTitle}</h4>
      </div>
      
      {/* Statistics Summary */}
      {statistics && (
        <div className="stats-row">
          <div className="stat-item">
            <span className="stat-label">{translations[language].currentHeight}</span>
            <span className="stat-value">{statistics.current?.toFixed(2) || statistics.min?.toFixed(2)} {translations[language].meters}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">{translations[language].maxHeight}</span>
            <span className="stat-value highlight-danger">{statistics.max?.toFixed(2)} {translations[language].meters}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">{translations[language].avgHeight}</span>
            <span className="stat-value">{statistics.mean?.toFixed(2)} {translations[language].meters}</span>
          </div>
        </div>
      )}
      
      {/* Chart */}
      <div className="chart-container">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="waveGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0d6efd" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#0d6efd" stopOpacity={0}/>
                </linearGradient>
              </defs>
              
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              
              <XAxis 
                dataKey="hour" 
                tick={{ fontSize: 11 }}
                tickLine={false}
                interval={2}
              />
              
              <YAxis 
                tick={{ fontSize: 11 }}
                tickLine={false}
                domain={[0, 'auto']}
                label={{ value: translations[language].heightLabel, angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              
              <Tooltip content={<CustomTooltip />} />
              
              {/* Safety threshold lines */}
              <ReferenceLine 
                y={1.0} 
                stroke="#28a745" 
                strokeDasharray="5 5" 
                label={{ value: translations[language].safeLimit, position: 'right', fontSize: 10, fill: '#28a745' }}
              />
              <ReferenceLine 
                y={2.5} 
                stroke="#dc3545" 
                strokeDasharray="5 5"
                label={{ value: translations[language].dangerLimit, position: 'right', fontSize: 10, fill: '#dc3545' }}
              />
              
              {/* Wave height area and line */}
              <Area 
                type="monotone" 
                dataKey="waveHeight" 
                fill="url(#waveGradient)" 
                stroke="none"
              />
              <Line
                type="monotone"
                dataKey="waveHeight"
                stroke="#0d6efd"
                strokeWidth={3}
                dot={{ fill: '#0d6efd', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, fill: '#0d6efd' }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="no-chart-data">
            <p>{translations[language].noForecastData}</p>
          </div>
        )}
      </div>
      
      {/* Legend */}
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-line safe"></span>
          <span>{translations[language].safeLegend}</span>
        </div>
        <div className="legend-item">
          <span className="legend-line caution"></span>
          <span>{translations[language].cautionLegend}</span>
        </div>
        <div className="legend-item">
          <span className="legend-line danger"></span>
          <span>{translations[language].dangerLegend}</span>
        </div>
      </div>
    </div>
  );
};

export default WaveForecastChart;
