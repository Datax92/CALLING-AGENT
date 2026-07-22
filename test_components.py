import asyncio
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient
from livekit.agents import llm

from app import app, CallSummary
from vllm_client import VLLMClient, VLLMStream
from rag_utils import RAGUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVLLMClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base_url = "http://test-vllm-server"
        self.client = VLLMClient(base_url=self.base_url)
        self.chat_ctx = MagicMock(spec=llm.ChatContext)
        self.chat_ctx.items = [
            MagicMock(role="user", text_content="Hello"),
            MagicMock(role="assistant", text_content="Hi there"),
        ]

    async def test_chat_returns_vllm_stream(self):
        stream = self.client.chat(chat_ctx=self.chat_ctx)
        self.assertIsInstance(stream, VLLMStream)

    async def test_stream_emits_chat_chunks(self):
        stream = VLLMStream(
            self.client,
            chat_ctx=self.chat_ctx,
            tools=[],
            conn_options=llm.DEFAULT_API_CONNECT_OPTIONS,
        )
        stream._event_ch = AsyncMock()

        # Mock the async generator
        async def mock_run():
            stream._event_ch.send_nowait = AsyncMock()
            await stream._run()

        await mock_run()
        self.assertTrue(stream._event_ch.send_nowait.called)

class TestRAGUtils(unittest.TestCase):
    def setUp(self):
        self.rag_utils = RAGUtils()
        self.query = "What services do you offer?"

    def test_filtered_lookup_returns_string(self):
        result = self.rag_utils.filtered_lookup(self.query)
        self.assertIsInstance(result, str)

    def test_filtered_lookup_truncates_to_500_tokens(self):
        # Create a mock knowledge base with a long entry
        long_entry = "word " * 600  # 600 words
        self.rag_utils.knowledge_base = [{"content": long_entry}]
        result = self.rag_utils.filtered_lookup(self.query)
        self.assertLessEqual(len(result.split()), 500)

class TestDashboard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(app)
        self.async_client = AsyncClient(app=app, base_url="http://test")
        self.test_summary = CallSummary(
            caller_number="+923001234567",
            business_name="Test Business",
            project_type="custom_software",
            agreed_price="",  # Not confirmed in initial call
            timeline="",  # Not confirmed in initial call
            email="test@example.com",
            phone_number="+923001234567",
            whatsapp_number="+923001234568",
            notes="Client is interested in custom software development for their business.",
            transcript_summary="Caller asked about our services. We explained our offerings and collected their contact details.",
            recording_url="https://r2.example.com/recordings/test-call.wav",
            call_duration=125.5  # 2 minutes 5 seconds
        )

    async def asyncTearDown(self):
        await self.async_client.aclose()

    async def test_receive_call_summary(self):
        # Mock the RAGUtils
        with patch('rag_utils.RAGUtils') as mock_rag_utils:
            mock_instance = mock_rag_utils.return_value
            mock_instance.filtered_lookup.return_value = "Test RAG snippet"

            response = self.client.post(
                "/webhook/call-summary",
                json=self.test_summary.dict(),
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("call_id", response.json())

    async def test_dashboard_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Deal Approvals", response.text)

    async def test_api_deals_endpoint(self):
        response = self.client.get("/api/deals")
        self.assertEqual(response.status_code, 200)
        self.assertIn("html", response.json())

if __name__ == "__main__":
    unittest.main()