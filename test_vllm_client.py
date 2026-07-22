import asyncio
import logging
import os
import sys

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm-client-test")

async def test_vllm_client():
    """Test the vLLM client integration."""
    logger.info("=== Testing vLLM Client ===")

    try:
        from vllm_client import VLLMClient
        from livekit.agents import llm

        # Create a vLLM client
        vllm_client = VLLMClient(base_url="http://localhost:8000")

        # Create a chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message("user", "What services do you offer?")
        chat_ctx.add_message("assistant", "We offer various services. Let me check our knowledge base.")

        # Test the streaming
        stream = vllm_client.chat(chat_ctx=chat_ctx)
        logger.info("Streaming response from vLLM client:")

        async for chunk in stream:
            if isinstance(chunk, llm.ChatChunk) and chunk.delta.content:
                logger.info(f"Received chunk: {chunk.delta.content}")

        logger.info("vLLM Client test PASSED")

    except Exception as e:
        logger.error(f"vLLM Client test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vllm_client())