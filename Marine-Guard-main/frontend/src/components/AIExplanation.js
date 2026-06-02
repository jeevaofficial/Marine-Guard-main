/**
 * AI Explanation Component
 * ========================
 * Displays GPT-4o generated explanations of marine conditions
 * 
 * Author: B.Tech AI&DS 
 * Date: 2026
 */

import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './AIExplanation.css';

const AIExplanation = ({ district, explanation, safetyStatus, aiProvider }) => {
  const { t } = useLanguage();
  
  // Format explanation text with proper line breaks
  const formatExplanation = (text) => {
    if (!text) return null;
    
    // Split by double newlines for paragraphs
    const paragraphs = text.split(/\n\n+/);
    
    return paragraphs.map((para, index) => {
      // Check if it's a header (starts with **)
      if (para.startsWith('**') && para.includes('**')) {
        const headerMatch = para.match(/\*\*(.+?)\*\*/);
        if (headerMatch) {
          const header = headerMatch[1];
          const rest = para.replace(/\*\*.+?\*\*/, '').trim();
          return (
            <div key={index} className="explanation-section">
              <h4 className="explanation-header">{header}</h4>
              <div className="explanation-text">{formatInlineText(rest)}</div>
            </div>
          );
        }
      }
      
      // Regular paragraph
      return (
        <div key={index} className="explanation-text">
          {formatInlineText(para)}
        </div>
      );
    });
  };
  
  // Format inline text (bold, lists, etc.)
  const formatInlineText = (text) => {
    // Handle bullet points
    const lines = text.split('\n');
    
    if (lines.length > 1 && lines.some(l => l.trim().startsWith('-') || l.trim().startsWith('•'))) {
      return (
        <ul className="explanation-list">
          {lines.filter(l => l.trim()).map((line, i) => (
            <li key={i}>{line.replace(/^[-•]\s*/, '').trim()}</li>
          ))}
        </ul>
      );
    }
    
    // Handle numbered lists
    if (lines.length > 1 && lines.some(l => /^\d+[.)\s]/.test(l.trim()))) {
      return (
        <ol className="explanation-list">
          {lines.filter(l => l.trim()).map((line, i) => (
            <li key={i}>{line.replace(/^\d+[.)\s]*/, '').trim()}</li>
          ))}
        </ol>
      );
    }
    
    return <p>{text}</p>;
  };

  return (
    <div className="card explanation-card">
      <div className="card-header">
        <h3 className="card-title">
          🤖 {t('aiAnalysis')}
        </h3>
        <span className={`ai-badge ${aiProvider === 'azure_openai' ? 'gpt' : 'fallback'}`}>
          {aiProvider === 'azure_openai' ? t('gptPowered') : t('ruleBased')}
        </span>
      </div>
      
      <div className="explanation-content">
        {explanation ? (
          <>
            <div className="explanation-body">
              {formatExplanation(explanation)}
            </div>
          </>
        ) : (
          <div className="no-explanation">
            <p>{t('noExplanation')}</p>
          </div>
        )}
      </div>
      
      <div className="explanation-footer">
        <div className="ai-info">
          <span className="info-icon">ℹ️</span>
          <span className="info-text">
            {aiProvider === 'azure_openai' 
              ? t('azureOpenAI')
              : t('ruleBasedSystem')}
          </span>
        </div>
      </div>
    </div>
  );
};

export default AIExplanation;
