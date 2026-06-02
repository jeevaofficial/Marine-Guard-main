/**
 * District Selector Component
 * ===========================
 * Dropdown to select from 14 Tamil Nadu coastal districts
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './DistrictSelector.css';

// All 14 coastal districts of Tamil Nadu
const DISTRICTS = [
  'Thiruvallur',
  'Chennai',
  'Kanchipuram',
  'Chengalpattu',
  'Villupuram',
  'Cuddalore',
  'Mayiladuthurai',
  'Nagapattinam',
  'Thanjavur',
  'Tiruvarur',
  'Ramanathapuram',
  'Thoothukudi',
  'Tirunelveli',
  'Kanniyakumari',
];

const DistrictSelector = ({ selectedDistrict, onDistrictChange }) => {
  const { t } = useLanguage();

  const handleChange = (e) => {
    onDistrictChange(e.target.value);
  };

  return (
    <div className="district-selector">
      <label className="selector-label">
        <span className="label-icon">📍</span>
        {t('selectDistrict')}
      </label>
      <div className="select-wrapper">
        <select
          className="select-input"
          value={selectedDistrict}
          onChange={handleChange}
        >
          {DISTRICTS.map((district) => (
            <option key={district} value={district}>
              {t(`districts.${district}`)}
            </option>
          ))}
        </select>
        <span className="select-arrow">▼</span>
      </div>
      <p className="selector-info">
        {t('selectDistrict')}: <strong>{t(`districts.${selectedDistrict}`)}</strong>
      </p>
    </div>
  );
};

export default DistrictSelector;
