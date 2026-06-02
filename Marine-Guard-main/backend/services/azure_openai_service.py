"""
Azure OpenAI GPT-4o Service for Marine Safety Explanations
===========================================================
This module integrates with Azure OpenAI GPT-4o to provide:
1. Natural language explanations of marine conditions
2. Safety recommendations for fishermen
3. Emergency warnings when conditions are dangerous

Prerequisites:
- Azure OpenAI resource with GPT-4o deployment
- API endpoint and key configured in .env file

Author: B.Tech AI&DS 
Date: 2026
"""

import os
import logging
from typing import Dict, Optional
from openai import AzureOpenAI

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)
from config.prompts import (
    get_system_prompt,
    get_explanation_prompt,
    format_explanation_prompt,
    format_emergency_prompt
)

# Set up logging
logger = logging.getLogger(__name__)


class AzureOpenAIService:
    """
    Service class for Azure OpenAI GPT-4o interactions.
    
    This class provides methods to:
    1. Generate explanations of marine safety conditions
    2. Create emergency warnings for dangerous conditions
    3. Provide actionable safety recommendations
    """
    
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Azure OpenAI service.
        
        Args:
            endpoint: Azure OpenAI endpoint URL (optional, uses env var if not provided)
            api_key: Azure OpenAI API key (optional, uses env var if not provided)
        """
        self.endpoint = endpoint or AZURE_OPENAI_ENDPOINT
        self.api_key = api_key or AZURE_OPENAI_API_KEY
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.api_version = AZURE_OPENAI_API_VERSION
        
        # Check if credentials are available
        self.is_configured = bool(self.endpoint and self.api_key)
        
        if self.is_configured:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            logger.info("Azure OpenAI Service initialized successfully")
        else:
            self.client = None
            logger.warning(
                "Azure OpenAI credentials not configured. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env file."
            )
    
    def generate_explanation(
        self,
        district: str,
        current_data: Dict,
        forecast_data: Dict,
        safety_status: str,
        forecast_hours: int = 24,
        language: str = 'en'
    ) -> str:
        """
        Generate natural language explanation of marine conditions.
        
        This method:
        1. Formats the prompt with actual data
        2. Calls GPT-4o API
        3. Returns explanation text
        
        Args:
            district: Name of the coastal district
            current_data: Dictionary with current conditions
            forecast_data: Dictionary with forecast statistics
            safety_status: One of 'Safe', 'Caution', 'Dangerous'
            forecast_hours: Number of hours in forecast period
            language: Language code ('en' or 'ta')
            
        Returns:
            Generated explanation text
        """
        if not self.is_configured:
            return self._generate_fallback_explanation(
                district, current_data, forecast_data, safety_status, language
            )
        
        # Format the prompt
        user_prompt = format_explanation_prompt(
            district=district,
            current_data=current_data,
            forecast_data=forecast_data,
            safety_status=safety_status,
            forecast_hours=forecast_hours,
            language=language
        )
        
        try:
            # Call GPT-4o with language-specific system prompt
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": get_system_prompt(language)},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"Generated explanation for {district} in {language}: {len(explanation)} chars")
            
            return explanation
            
        except Exception as e:
            logger.error(f"GPT-4o API call failed: {e}")
            return self._generate_fallback_explanation(
                district, current_data, forecast_data, safety_status, language
            )
    
    def generate_emergency_warning(
        self,
        district: str,
        wave_height: float,
        wind_speed: float,
        duration: int
    ) -> str:
        """
        Generate emergency warning message for dangerous conditions.
        
        Args:
            district: Name of the coastal district
            wave_height: Current/predicted wave height in meters
            wind_speed: Current/predicted wind speed in m/s
            duration: Expected duration of dangerous conditions in hours
            
        Returns:
            Emergency warning text
        """
        if not self.is_configured:
            return self._generate_fallback_emergency(
                district, wave_height, wind_speed, duration
            )
        
        # Format emergency prompt
        user_prompt = format_emergency_prompt(
            district=district,
            wave_height=wave_height,
            wind_speed=wind_speed,
            duration=duration
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": get_system_prompt('en')},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.5  # Lower temperature for safety-critical content
            )
            
            warning = response.choices[0].message.content
            logger.info(f"Generated emergency warning for {district}")
            
            return warning
            
        except Exception as e:
            logger.error(f"Emergency warning generation failed: {e}")
            return self._generate_fallback_emergency(
                district, wave_height, wind_speed, duration
            )
    
    def _generate_fallback_explanation(
        self,
        district: str,
        current_data: Dict,
        forecast_data: Dict,
        safety_status: str,
        language: str = 'en'
    ) -> str:
        """
        Generate rule-based fallback explanation when GPT-4o is unavailable.
        
        This ensures the system works even without Azure OpenAI configured.
        """
        wave_height = current_data.get("wave_height", 0)
        wind_speed = current_data.get("wind_speed", 0)
        max_wave = forecast_data.get("max_wave_height", wave_height)
        
        if language == 'ta':
            # Tamil fallback
            if safety_status == "Safe":
                explanation = f"""
**{district} கடல் நிலை பகுப்பாய்வு**

தற்போதைய கடல் நிலைமை மீன்பிடித்தலுக்கு **பாதுகாப்பானது**.

**பாதுகாப்பானதற்கான காரணங்கள்:**
- தற்போதைய அலை உயரம் {wave_height:.2f}மீ, பாதுகாப்பு வரம்பான 1.0மீக்கு கீழ் உள்ளது
- காற்றின் வேகம் {wind_speed:.1f} m/s, மிதமான அளவில் உள்ளது
- அதிகபட்ச அலை உயரம் {max_wave:.2f}மீ, பாதுகாப்பான வரம்புக்குள் உள்ளது

**பரிந்துரைகள்:**
- கடலோர மற்றும் நடுக்கடல் மீன்பிடித்தலுக்கு சாதகமான நிலைமைகள்
- வழக்கமான பாதுகாப்பு முன்னெச்சரிக்கைகளை பின்பற்றவும்
- நீண்ட பயணங்களுக்கு வானிலை நிலைமைகளை கண்காணிக்கவும்
- தொடர்பு சாதனங்களை எடுத்துச் செல்லவும்
"""
            elif safety_status == "Caution":
                explanation = f"""
**{district} கடல் நிலை பகுப்பாய்வு**

மீன்பிடித்தலுக்கு **எச்சரிக்கை** தேவை.

**எச்சரிக்கை அறிவுறுத்தப்படுவதற்கான காரணங்கள்:**
- தற்போதைய அலை உயரம் {wave_height:.2f}மீ, 1.0மீ மற்றும் 2.5மீக்கு இடையில் உள்ளது
- காற்றின் வேகம் {wind_speed:.1f} m/s, கடலில் சீற்றத்தை ஏற்படுத்தலாம்
- அதிகபட்ச அலை உயரம் {max_wave:.2f}மீ வரை உயரலாம்

**பரிந்துரைகள்:**
- கடலில் அதிக தூரம் செல்வதை தவிர்க்கவும்
- சிறிய படகுகள் துறைமுகத்தில் இருக்க வேண்டும்
- அனுபவம் வாய்ந்த மீனவர்கள் மட்டும் குறுகிய பயணங்களை மேற்கொள்ளலாம்
- வானிலை புதுப்பிப்புகளை தொடர்ந்து கண்காணிக்கவும்
"""
            else:  # Dangerous
                explanation = f"""
**{district} கடல் நிலை பகுப்பாய்வு**

**ஆபத்து - மீன்பிடித்தல் தடை செய்யப்படுகிறது**

**ஆபத்தானதற்கான காரணங்கள்:**
- அலை உயரம் {wave_height:.2f}மீ, மிகவும் உயர்ந்த நிலையில் உள்ளது (2.5மீக்கு மேல்)
- காற்றின் வேகம் {wind_speed:.1f} m/s, மிகவும் ஆபத்தான நிலையில் உள்ளது
- அதிகபட்ச அலை உயரம் {max_wave:.2f}மீ வரை உயரும்

**கட்டாய அறிவுறுத்தல்கள்:**
- அனைத்து மீன்பிடித்தலும் ரத்து செய்யப்பட வேண்டும்
- துறைமுகத்தில் இருக்கவும்
- படகுகளை பாதுகாப்பாக கட்டி வைக்கவும்
- உயிர் பாதுகாப்பு சாதனங்களை தயார் நிலையில் வைக்கவும்
- உடனடி எச்சரிக்கைகளுக்கு வானிலை அறிக்கைகளை கவனிக்கவும்
"""
        else:
            # English fallback
            if safety_status == "Safe":
                explanation = f"""
**Sea Conditions Analysis for {district}**

The sea conditions are currently **SAFE** for fishing activities.

**Why it's Safe:**
- Current wave height of {wave_height:.2f}m is below the 1.0m safety threshold
- Wind speeds at {wind_speed:.1f} m/s are moderate
- Maximum predicted wave height of {max_wave:.2f}m remains within safe limits

**Recommendations:**
- Favorable conditions for near-shore and offshore fishing
- Still maintain standard safety precautions
- Monitor conditions if planning extended trips
- Carry communication devices
"""
            elif safety_status == "Caution":
                explanation = f"""
**Sea Conditions Analysis for {district}**

Sea conditions require **CAUTION** for fishing activities.

**Why Caution is Advised:**
- Current wave height of {wave_height:.2f}m is between 1.0m and 2.5m
- Wind speeds at {wind_speed:.1f} m/s may cause rough conditions
- Maximum predicted wave height could reach {max_wave:.2f}m

**Recommendations:**
- Avoid venturing far from shore
- Small boats should stay in harbor
- Only experienced fishermen should consider short trips
- Keep monitoring weather updates
- Ensure emergency equipment is ready
"""
            else:  # Dangerous
                explanation = f"""
**Sea Conditions Analysis for {district}**

⚠️ **DANGEROUS CONDITIONS - DO NOT VENTURE INTO SEA** ⚠️

**Why it's Dangerous:**
- Wave height of {wave_height:.2f}m exceeds the 2.5m danger threshold
- Wind speeds at {wind_speed:.1f} m/s create hazardous conditions
- Maximum predicted wave height of {max_wave:.2f}m poses severe risk

**Mandatory Precautions:**
- ALL fishing activities must be suspended
- Do not launch any vessels
- Secure boats and fishing equipment
- Stay away from beaches and coastal areas
- Wait for official all-clear announcement
"""
        
        return explanation.strip()
    
    def _generate_fallback_emergency(
        self,
        district: str,
        wave_height: float,
        wind_speed: float,
        duration: int
    ) -> str:
        """
        Generate rule-based emergency warning when GPT-4o is unavailable.
        """
        wind_knots = wind_speed * 1.944
        
        return f"""
🚨 **EMERGENCY MARINE SAFETY ALERT** 🚨

**Location:** {district} Coast, Tamil Nadu
**Issued:** {__import__('datetime').datetime.now().strftime('%d %b %Y, %I:%M %p')}

**DANGEROUS SEA CONDITIONS DETECTED**

Current measurements:
- Wave Height: {wave_height:.2f} meters (EXCEEDS SAFE LIMIT)
- Wind Speed: {wind_speed:.1f} m/s ({wind_knots:.1f} knots)
- Expected Duration: {duration} hours

**IMMEDIATE ACTIONS REQUIRED:**

1. DO NOT venture into the sea under any circumstances
2. All boats must return to harbor immediately
3. Secure vessels and fishing equipment
4. Move away from beaches and low-lying coastal areas
5. Alert fellow fishermen who may not have received this warning

**Emergency Contacts:**
- Coast Guard: 1554
- District Emergency: 108
- Fisheries Department: Contact local office

**This warning remains in effect for the next {duration} hours.**

Stay safe. Protect your life first.
""".strip()
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI service is configured and available."""
        return self.is_configured
    
    def test_connection(self) -> Dict:
        """
        Test the Azure OpenAI connection.
        
        Returns:
            Dictionary with connection status and details
        """
        if not self.is_configured:
            return {
                "status": "not_configured",
                "message": "Azure OpenAI credentials not set in .env file",
                "required": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
            }
        
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": "Hello, respond with 'OK' only."}
                ],
                max_tokens=10
            )
            
            return {
                "status": "connected",
                "message": "Azure OpenAI connection successful",
                "deployment": self.deployment,
                "response": response.choices[0].message.content
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "deployment": self.deployment
            }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = AzureOpenAIService()
    
    # Test connection
    print("\nTesting Azure OpenAI connection...")
    status = service.test_connection()
    print(f"Status: {status}")
    
    # Test explanation generation (will use fallback if not configured)
    print("\nGenerating test explanation...")
    current_data = {
        "wave_height": 1.5,
        "wave_period": 8.0,
        "wind_speed": 12.0,
        "wind_direction": 180,
        "temperature": 28.0,
        "humidity": 80,
        "pressure": 1010
    }
    
    forecast_data = {
        "max_wave_height": 2.0,
        "avg_wave_height": 1.6,
        "max_wind_speed": 15.0
    }
    
    explanation = service.generate_explanation(
        district="Chennai",
        current_data=current_data,
        forecast_data=forecast_data,
        safety_status="Caution"
    )
    
    print("\nGenerated Explanation:")
    print(explanation)
