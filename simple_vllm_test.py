import asyncio
import logging
import os
import sys
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-vllm-test")

async def test_mock_server():
    """Test the mock vLLM server directly."""
    logger.info("=== Testing Mock vLLM Server ===")

    try:
        # Test the health endpoint
        response = requests.get("http://localhost:8001/health")
        logger.info(f"Health check: {response.status_code} - {response.json()}")

        # Test the chat completions endpoint
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": "What services do you offer?"
        }

        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            headers=headers,
            json={"prompt": "What services do you offer?"},
            stream=True
        )

        logger.info(f"Chat completions status: {response.status_code}")

        # Read the streaming response
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_data = json.loads(decoded_line[6:])
                    logger.info(f"Received chunk: {json_data.get('text', '')}")

        logger.info("Mock vLLM Server test PASSED")

    except Exception as e:
        logger.error(f"Mock vLLM Server test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mock_server())