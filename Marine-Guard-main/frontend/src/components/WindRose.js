/**
 * Wind Rose Component
 * ===================
 * Displays wind direction distribution in a circular chart
 * Shows prevailing wind patterns for the selected district
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React, { useMemo } from 'react';
import { PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { useLanguage } from '../contexts/LanguageContext';
import './WindRose.css';

const WindRose = ({ forecastData }) => {
  const { t } = useLanguage();

  // Process wind data into 8 directional sectors (N, NE, E, SE, S, SW, W, NW)
  const windRoseData = useMemo(() => {
    if (!forecastData) {
      return null;
    }

    // Check if we have the raw forecast data with hourly details
    const hourlyData = forecastData.hourly_data;
    
    if (!hourlyData || !Array.isArray(hourlyData) || hourlyData.length === 0) {
      console.log('No hourly wind data available in forecast:', forecastData);
      return null;
    }

    // Verify we have wind direction and speed data
    const hasWindData = hourlyData.some(hour => 
      hour.wind_direction !== undefined && 
      hour.wind_speed !== undefined
    );
    
    if (!hasWindData) {
      console.log('Hourly data exists but no wind_direction/wind_speed fields');
      return null;
    }

    // Define 8 wind directions
    const directions = [
      { name: 'N', label: t('windRose.north'), min: 337.5, max: 22.5 },
      { name: 'NE', label: t('windRose.northeast'), min: 22.5, max: 67.5 },
      { name: 'E', label: t('windRose.east'), min: 67.5, max: 112.5 },
      { name: 'SE', label: t('windRose.southeast'), min: 112.5, max: 157.5 },
      { name: 'S', label: t('windRose.south'), min: 157.5, max: 202.5 },
      { name: 'SW', label: t('windRose.southwest'), min: 202.5, max: 247.5 },
      { name: 'W', label: t('windRose.west'), min: 247.5, max: 292.5 },
      { name: 'NW', label: t('windRose.northwest'), min: 292.5, max: 337.5 }
    ];

    // Initialize counters
    const directionCounts = directions.map(d => ({ 
      ...d, 
      count: 0, 
      totalSpeed: 0,
      maxSpeed: 0 
    }));

    // Process hourly wind data
    hourlyData.forEach(hour => {
      const direction = hour.wind_direction;
      const speed = hour.wind_speed;

      if (direction === undefined || speed === undefined) {
        return;
      }

      // Find which sector this direction belongs to
      let sectorIndex = -1;
      
      // Handle North sector (wraps around 0/360)
      if (direction >= 337.5 || direction < 22.5) {
        sectorIndex = 0;
      } else {
        sectorIndex = directions.findIndex((d, idx) => 
          idx > 0 && direction >= d.min && direction < d.max
        );
      }

      if (sectorIndex >= 0) {
        directionCounts[sectorIndex].count++;
        directionCounts[sectorIndex].totalSpeed += speed;
        directionCounts[sectorIndex].maxSpeed = Math.max(
          directionCounts[sectorIndex].maxSpeed, 
          speed
        );
      }
    });

    // Calculate percentages and average speeds
    const totalPoints = hourlyData.filter(h => 
      h.wind_direction !== undefined && h.wind_speed !== undefined
    ).length;
    
    if (totalPoints === 0) {
      return null;
    }
    
    return directionCounts.map(d => ({
      direction: d.label,
      shortName: d.name,
      frequency: ((d.count / totalPoints) * 100).toFixed(1),
      avgSpeed: d.count > 0 ? (d.totalSpeed / d.count).toFixed(1) : 0,
      maxSpeed: d.maxSpeed.toFixed(1),
      count: d.count
    }));
  }, [forecastData, t]);

  // Find prevailing wind direction
  const prevailingWind = useMemo(() => {
    if (!windRoseData) return null;
    return windRoseData.reduce((prev, current) => 
      parseFloat(current.frequency) > parseFloat(prev.frequency) ? current : prev
    );
  }, [windRoseData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="wind-rose-tooltip">
          <p className="tooltip-direction">{data.direction}</p>
          <p className="tooltip-item">
            <strong>{t('windRose.frequency')}:</strong> {data.frequency}%
          </p>
          <p className="tooltip-item">
            <strong>{t('windRose.avgSpeed')}:</strong> {data.avgSpeed} m/s
          </p>
          <p className="tooltip-item">
            <strong>{t('windRose.maxSpeed')}:</strong> {data.maxSpeed} m/s
          </p>
          <p className="tooltip-item">
            <strong>{t('windRose.observations')}:</strong> {data.count}
          </p>
        </div>
      );
    }
    return null;
  };

  if (!windRoseData) {
    return (
      <div className="wind-rose-card-content">
        <div className="chart-subheader">
          <h4 className="chart-subtitle">🧭 {t('windRose.title')}</h4>
        </div>
        <div className="no-data">
          <p>{t('windRose.noData')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="wind-rose-card-content">
      <div className="chart-subheader">
        <h4 className="chart-subtitle">🧭 {t('windRose.title')}</h4>
      </div>

      {/* Prevailing Wind Info */}
      {prevailingWind && (
        <div className="prevailing-wind-info">
          <div className="prevailing-label">{t('windRose.prevailingWind')}:</div>
          <div className="prevailing-details">
            <span className="prevailing-direction">{prevailingWind.direction}</span>
            <span className="prevailing-stats">
              {prevailingWind.frequency}% • {prevailingWind.avgSpeed} m/s
            </span>
          </div>
        </div>
      )}

      {/* Wind Rose Chart */}
      <div className="wind-rose-chart">
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={windRoseData}>
            <PolarGrid stroke="#cbd5e1" strokeDasharray="3 3" />
            <PolarAngleAxis 
              dataKey="shortName" 
              tick={{ fill: '#64748b', fontSize: 14, fontWeight: 600 }}
            />
            <PolarRadiusAxis 
              angle={90} 
              domain={[0, 'auto']} 
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <Radar 
              name={t('windRose.frequency')}
              dataKey="frequency" 
              stroke="#1e88e5" 
              fill="#1e88e5" 
              fillOpacity={0.6}
              strokeWidth={2}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{ paddingTop: '10px' }}
              iconType="circle"
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Wind Speed Legend */}
      <div className="wind-legend">
        <div className="legend-title">{t('windRose.speedCategories')}:</div>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#10b981' }}></span>
            <span>{t('windRose.light')} (&lt; 5 m/s)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#f59e0b' }}></span>
            <span>{t('windRose.moderate')} (5-10 m/s)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#ef4444' }}></span>
            <span>{t('windRose.strong')} (&gt; 10 m/s)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WindRose;
