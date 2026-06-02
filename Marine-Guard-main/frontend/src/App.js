/**
 * Main Application Component
 * ==========================
 * Root component for the Marine Safety Forecasting System
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { LanguageProvider } from './contexts/LanguageContext';
import Dashboard from './components/Dashboard';
import LanguageToggle from './components/LanguageToggle';
import './App.css';

function App() {
  return (
    <LanguageProvider>
      <div className="App">
        {/* Language Toggle */}
        <LanguageToggle />
        
        {/* Header */}
        <AppHeader />
        
        {/* Main Dashboard */}
        <main className="app-main">
          <Dashboard />
        </main>
        
        {/* Footer */}
        <footer className="app-footer">
          <p>© 2026 Marine Safety Forecasting System | B.Tech AI&DS </p>
        </footer>
      </div>
    </LanguageProvider>
  );
}

// Separate Header Component with language support
function AppHeader() {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="logo-section">
          <span className="logo-icon">🌊</span>
          <div className="title-section">
            <HeaderTitle />
          </div>
        </div>
      </div>
      </header>
  );
}

// Header Title with i18n support
function HeaderTitle() {
  const { useLanguage } = require('./contexts/LanguageContext');
  const { t } = useLanguage();
  
  return (
    <>
      <h1>{t('appTitle')}</h1>
      <p className="subtitle">{t('appSubtitle')} • AI-Powered Predictions</p>
    </>
  );
}

export default App;
