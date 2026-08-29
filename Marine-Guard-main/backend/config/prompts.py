"""
GPT-4o Prompt Templates for Marine Safety Explanations
=======================================================
This module contains carefully crafted prompts for Groq AI GPT-4o
to generate human-readable explanations of marine safety conditions.

The prompts are designed to:
1. Be context-aware (include actual forecast data)
2. Provide actionable safety recommendations
3. Be accessible to fishermen and coastal communities
4. Support Tamil Nadu specific context

Author: B.Tech AI&DS  Project
Date: 2026
"""

# =============================================================================
# SYSTEM PROMPT - DEFINES THE AI'S ROLE AND BEHAVIOR
# =============================================================================

SYSTEM_PROMPT_ENGLISH = """You are a Marine Safety Expert AI assistant for Tamil Nadu's coastal regions in India.
Your role is to explain sea conditions and provide safety recommendations to fishermen and coastal communities.

Guidelines for your responses:
1. Be clear, concise, and use simple English language
2. Always prioritize human safety
3. Reference specific data values (wave height, wind speed, etc.)
4. Provide practical, actionable advice
5. Consider local fishing community practices
6. Be respectful of traditional fishing knowledge
7. Include time-specific warnings when relevant
8. Mention any approaching weather systems

Your audience includes:
- Traditional fishermen with varying education levels
- Coastal tourism operators
- Maritime authorities
- Emergency response teams

Always err on the side of caution when safety is concerned."""

SYSTEM_PROMPT_TAMIL = """You are a Marine Safety Expert AI assistant for Tamil Nadu's coastal regions in India.
Your role is to explain sea conditions and provide safety recommendations to fishermen and coastal communities.

IMPORTANT: You MUST respond in Tamil (தமிழ்) language. Write your entire explanation in Tamil script.

Guidelines for your responses:
1. Write everything in Tamil language using Tamil script
2. Be clear, concise, and use simple Tamil language
3. Always prioritize human safety
4. Reference specific data values (wave height, wind speed, etc.)
5. Provide practical, actionable advice in Tamil
6. Consider local fishing community practices
7. Be respectful of traditional fishing knowledge
8. Include time-specific warnings when relevant
9. Mention any approaching weather systems

Your audience includes:
- Traditional Tamil Nadu fishermen with varying education levels
- Coastal tourism operators
- Maritime authorities
- Emergency response teams

Always err on the side of caution when safety is concerned.
Remember: Respond ONLY in Tamil language."""

def get_system_prompt(language='en'):
    """Get system prompt based on language."""
    return SYSTEM_PROMPT_TAMIL if language == 'ta' else SYSTEM_PROMPT_ENGLISH

# =============================================================================
# EXPLANATION PROMPT TEMPLATE
# =============================================================================

EXPLANATION_PROMPT_ENGLISH = """Based on the following marine and weather forecast data for {district} district, Tamil Nadu coast:

**Current Conditions:**
- Wave Height: {current_wave_height:.2f} meters
- Wave Period: {current_wave_period:.1f} seconds
- Wind Speed: {current_wind_speed:.1f} m/s ({current_wind_speed_knots:.1f} knots)
- Wind Direction: {current_wind_direction}°
- Temperature: {current_temperature:.1f}°C
- Humidity: {current_humidity:.0f}%
- Surface Pressure: {current_pressure:.1f} hPa

**Forecast for Next {forecast_hours} Hours:**
- Maximum Predicted Wave Height: {max_wave_height:.2f} meters
- Average Predicted Wave Height: {avg_wave_height:.2f} meters
- Maximum Predicted Wind Speed: {max_wind_speed:.1f} m/s

**Safety Classification: {safety_status}**

Please provide:
1. A brief explanation (2-3 sentences) of WHY the sea conditions are classified as {safety_status}
2. Specific safety recommendations for fishermen
3. Any additional warnings or advice for the next {forecast_hours} hours

Keep your response under 200 words and use simple language."""

EXPLANATION_PROMPT_TAMIL = """Based on the following marine and weather forecast data for {district} district, Tamil Nadu coast:

**Current Conditions:**
- Wave Height: {current_wave_height:.2f} meters
- Wave Period: {current_wave_period:.1f} seconds
- Wind Speed: {current_wind_speed:.1f} m/s ({current_wind_speed_knots:.1f} knots)
- Wind Direction: {current_wind_direction}°
- Temperature: {current_temperature:.1f}°C
- Humidity: {current_humidity:.0f}%
- Surface Pressure: {current_pressure:.1f} hPa

**Forecast for Next {forecast_hours} Hours:**
- Maximum Predicted Wave Height: {max_wave_height:.2f} meters
- Average Predicted Wave Height: {avg_wave_height:.2f} meters
- Maximum Predicted Wind Speed: {max_wind_speed:.1f} m/s

**Safety Classification: {safety_status}**

Please provide your response in Tamil (தமிழ்) language:
1. கடல் நிலைமைகள் {safety_status} என வகைப்படுத்தப்பட்டதற்கான காரணம் (2-3 வாக்கியங்கள்)
2. மீனவர்களுக்கான குறிப்பிட்ட பாதுகாப்பு பரிந்துரைகள்
3. அடுத்த {forecast_hours} மணிநேரத்திற்கான கூடுதல் எச்சரிக்கைகள் அல்லது ஆலோசனைகள்

முழு பதிலும் தமிழில் இருக்க வேண்டும். 200 வார்த்தைகளுக்குள் எளிய தமிழில் எழுதவும்."""

def get_system_prompt(language='en'):
    """Get system prompt based on language
    
    Args:
        language (str): Language code ('en' or 'ta')
        
    Returns:
        str: System prompt in the specified language
    """
    if language == 'ta':
        return SYSTEM_PROMPT_TAMIL
    return SYSTEM_PROMPT_ENGLISH

def get_explanation_prompt(language='en'):
    """Get explanation prompt template based on language
    
    Args:
        language (str): Language code ('en' or 'ta')
        
    Returns:
        str: Explanation prompt template in the specified language
    """
    if language == 'ta':
        return EXPLANATION_PROMPT_TAMIL
    return EXPLANATION_PROMPT_ENGLISH

# =============================================================================
# DETAILED ANALYSIS PROMPT
# =============================================================================

DETAILED_ANALYSIS_PROMPT_TEMPLATE = """Analyze the following marine forecast data for {district}, Tamil Nadu:

**Hourly Wave Height Forecast (next 24 hours):**
{hourly_wave_heights}

**Hourly Wind Speed Forecast (next 24 hours):**
{hourly_wind_speeds}

**Climate Conditions:**
- Temperature Range: {temp_min:.1f}°C to {temp_max:.1f}°C
- Humidity Range: {humidity_min:.0f}% to {humidity_max:.0f}%
- Pressure Trend: {pressure_trend}
- Precipitation Expected: {precipitation_expected}

**Overall Safety Assessment: {safety_status}**

Provide a detailed analysis covering:
1. Best time windows for fishing activities (if any)
2. Expected sea state changes throughout the day
3. Specific dangers to watch for
4. Recommendations for different types of vessels (small boats vs larger boats)
5. Emergency preparedness advice

Be specific with times and measurements."""

# =============================================================================
# EMERGENCY WARNING PROMPT
# =============================================================================

EMERGENCY_WARNING_PROMPT_TEMPLATE = """URGENT MARINE SAFETY ALERT for {district}, Tamil Nadu

**DANGEROUS CONDITIONS DETECTED:**
- Wave Height: {wave_height:.2f} meters (EXCEEDS SAFE LIMIT)
- Wind Speed: {wind_speed:.1f} m/s ({wind_speed_knots:.1f} knots)
- Conditions Expected: {duration} hours

Generate an emergency warning message that:
1. Clearly states the danger level
2. Lists immediate actions to take
3. Provides emergency contact guidance
4. Warns about specific risks (capsizing, drowning, etc.)
5. Advises when conditions may improve

This message will be sent to fishing communities. Make it urgent but not panic-inducing."""

# =============================================================================
# COMPARATIVE ANALYSIS PROMPT
# =============================================================================

COMPARISON_PROMPT_TEMPLATE = """Compare marine conditions across multiple Tamil Nadu coastal districts:

{district_data}

Based on this comparison:
1. Which districts are safest for fishing today?
2. Which districts should be avoided?
3. Are there any regional weather patterns affecting the coast?
4. Provide a brief summary for maritime authorities.

Keep the response concise and actionable."""

# =============================================================================
# HELPER FUNCTION FOR PROMPT FORMATTING
# =============================================================================

def format_explanation_prompt(
    district: str,
    current_data: dict,
    forecast_data: dict,
    safety_status: str,
    forecast_hours: int = 24,
    language: str = 'en'
) -> str:
    """
    Format the explanation prompt with actual data values.
    
    Args:
        district: Name of the coastal district
        current_data: Dictionary containing current weather/marine conditions
        forecast_data: Dictionary containing forecast statistics
        safety_status: One of 'Safe', 'Caution', or 'Dangerous'
        forecast_hours: Number of hours in the forecast period
        language: Language code ('en' or 'ta')
    
    Returns:
        Formatted prompt string ready for GPT-4o
    """
    # Convert wind speed from m/s to knots (1 m/s = 1.944 knots)
    wind_speed_knots = current_data.get("wind_speed", 0) * 1.944
    
    # Get the appropriate prompt template for the language
    prompt_template = get_explanation_prompt(language)
    
    return prompt_template.format(
        district=district,
        current_wave_height=current_data.get("wave_height", 0),
        current_wave_period=current_data.get("wave_period", 0),
        current_wind_speed=current_data.get("wind_speed", 0),
        current_wind_speed_knots=wind_speed_knots,
        current_wind_direction=current_data.get("wind_direction", 0),
        current_temperature=current_data.get("temperature", 25),
        current_humidity=current_data.get("humidity", 70),
        current_pressure=current_data.get("pressure", 1013),
        forecast_hours=forecast_hours,
        max_wave_height=forecast_data.get("max_wave_height", 0),
        avg_wave_height=forecast_data.get("avg_wave_height", 0),
        max_wind_speed=forecast_data.get("max_wind_speed", 0),
        safety_status=safety_status
    )


def format_emergency_prompt(
    district: str,
    wave_height: float,
    wind_speed: float,
    duration: int
) -> str:
    """
    Format emergency warning prompt for dangerous conditions.
    
    Args:
        district: Name of the coastal district
        wave_height: Current/predicted wave height in meters
        wind_speed: Current/predicted wind speed in m/s
        duration: Expected duration of dangerous conditions in hours
    
    Returns:
        Formatted emergency prompt string
    """
    wind_speed_knots = wind_speed * 1.944
    
    return EMERGENCY_WARNING_PROMPT_TEMPLATE.format(
        district=district,
        wave_height=wave_height,
        wind_speed=wind_speed,
        wind_speed_knots=wind_speed_knots,
        duration=duration
    )
