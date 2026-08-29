#!/usr/bin/env python
"""Test Groq service configuration and connectivity."""

from services.groq_service import GroqService
from config.settings import GROQ_API_KEY
import json

print("=" * 60)
print("GROQ SERVICE TEST")
print("=" * 60)

# Check API key
print(f"\n1. API Key Status:")
print(f"   Configured: {bool(GROQ_API_KEY)}")
print(f"   Key (first 20 chars): {GROQ_API_KEY[:20] if GROQ_API_KEY else 'NOT SET'}...")

# Initialize service
service = GroqService()
print(f"\n2. Service Initialization:")
print(f"   Is Configured: {service.is_configured}")
print(f"   Is Available: {service.is_available()}")
print(f"   Deployment: {service.deployment}")

# Test connection
print(f"\n3. Connection Test:")
result = service.test_connection()
print(f"   Result: {json.dumps(result, indent=2)}")

# Try to generate explanation
print(f"\n4. Generate Explanation Test:")
try:
    explanation = service.generate_explanation(
        district="Chennai",
        current_data={"wave_height": 1.5, "wind_speed": 10},
        forecast_data={"max_wave_height": 2.0},
        safety_status="Safe",
        language="en"
    )
    print(f"   Success!")
    print(f"   Explanation (first 100 chars): {explanation[:100]}...")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {str(e)}")

print("\n" + "=" * 60)
