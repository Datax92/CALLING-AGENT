import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard-test")

DASHBOARD_WEBHOOK_URL = os.getenv(
    "DASHBOARD_WEBHOOK_URL", "http://localhost:8000/webhook/call-summary"
)

def send_test_call_summary():
    """Send a test call summary to the dashboard webhook."""
    logger.info("=== Testing Dashboard Webhook ===")

    # Create a test payload with all the new fields
    payload = {
        "caller_number": "+923001234567",
        "business_name": "Test Business",
        "project_type": "custom_software",
        "agreed_price": "",  # Not confirmed in initial call
        "timeline": "",  # Not confirmed in initial call
        "email": "test@example.com",
        "phone_number": "+923001234567",
        "whatsapp_number": "+923001234568",
        "notes": "Client is interested in custom software development for their business.",
        "transcript_summary": "Caller asked about our services. We explained our offerings and collected their contact details.",
        "recording_url": "https://r2.example.com/recordings/test-call.wav",
        "call_duration": 125.5  # 2 minutes 5 seconds
    }

    try:
        # Send the payload to the webhook
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DASHBOARD_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            logger.info(f"Webhook response: {response}")
            logger.info("Dashboard webhook test PASSED")
            return response.get("call_id")

    except Exception as e:
        logger.error(f"Dashboard webhook test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    call_id = send_test_call_summary()
    if call_id:
        logger.info(f"Call ID: {call_id}")
        logger.info("Check the dashboard at http://localhost:8000 to see the call details")