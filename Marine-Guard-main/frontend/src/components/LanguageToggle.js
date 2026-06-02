/**
 * Language Toggle Component
 * =========================
 * Button to switch between English and Tamil
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './LanguageToggle.css';

const LanguageToggle = () => {
  const { language, toggleLanguage } = useLanguage();

  return (
    <button 
      className="language-toggle"
      onClick={toggleLanguage}
      title={language === 'en' ? 'Switch to Tamil' : 'Switch to English'}
    >
      <span className="lang-icon">🌐</span>
      <span className="lang-text">
        {language === 'en' ? 'த' : 'EN'}
      </span>
    </button>
  );
};

export default LanguageToggle;
