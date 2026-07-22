import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Any

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local-test-runner")

class LocalTestRunner:
    def __init__(self):
        self.test_results = []

    async def test_rag_utils(self) -> None:
        """Test the RAG utilities."""
        logger.info("=== Testing RAG Utilities ===")
        try:
            from rag_utils import RAGUtils, get_rag_confidence

            rag_utils = RAGUtils()

            # Test filtered lookup
            test_queries = [
                "What services do you offer?",
                "How is a quotation prepared?",
                "What about SEO?",
                "This should not match anything"
            ]

            for query in test_queries:
                result = rag_utils.filtered_lookup(query)
                confidence = get_rag_confidence(query)

                logger.info(f"Query: '{query}'")
                logger.info(f"Confidence: {confidence:.2f}")

                if result:
                    logger.info(f"Result length: {len(result)} characters")
                    logger.info("Result preview: %s...", result[:100])
                else:
                    logger.info("No result returned")

                logger.info("---")

            self.test_results.append({"name": "RAG Utilities", "status": "PASSED"})
            logger.info("RAG Utilities test PASSED")

        except Exception as e:
            logger.error(f"RAG Utilities test FAILED: {e}")
            self.test_results.append({"name": "RAG Utilities", "status": "FAILED", "error": str(e)})

    async def test_latency_metrics(self) -> None:
        """Test the latency metrics logging."""
        logger.info("=== Testing Latency Metrics ===")
        try:
            from latency_metrics import log_stage

            log_stage("test_stage", 123.45, test_field="test_value")
            logger.info("Latency metric logged - check logs for output")

            self.test_results.append({"name": "Latency Metrics", "status": "PASSED"})
            logger.info("Latency Metrics test PASSED")

        except Exception as e:
            logger.error(f"Latency Metrics test FAILED: {e}")
            self.test_results.append({"name": "Latency Metrics", "status": "FAILED", "error": str(e)})

    async def test_gpu_scheduler(self) -> None:
        """Test the GPU scheduler."""
        logger.info("=== Testing GPU Scheduler ===")
        try:
            from gpu_scheduler import scheduler

            # Just verify the scheduler can be imported and initialized
            logger.info("GPU Scheduler imported successfully")

            self.test_results.append({"name": "GPU Scheduler", "status": "PASSED"})
            logger.info("GPU Scheduler test PASSED")

        except Exception as e:
            logger.error(f"GPU Scheduler test FAILED: {e}")
            self.test_results.append({"name": "GPU Scheduler", "status": "FAILED", "error": str(e)})

    async def test_sip_bridge(self) -> None:
        """Test the SIP bridge configuration."""
        logger.info("=== Testing SIP Bridge Configuration ===")
        try:
            # Just verify the configuration files exist and are valid JSON
            import json

            # Check dispatch-rule.json
            try:
                with open('dispatch-rule.json', 'r', encoding='utf-8') as f:
                    dispatch_rule = json.load(f)
                    logger.info(f"Dispatch rule loaded: {dispatch_rule.get('name')}")
            except Exception as e:
                logger.error(f"Error loading dispatch-rule.json: {e}")
                raise

            # Check inbound-trunk.json
            try:
                with open('inbound-trunk.json', 'r', encoding='utf-8') as f:
                    inbound_trunk = json.load(f)
                    logger.info(f"Inbound trunk loaded: {inbound_trunk.get('trunk', {}).get('name')}")
            except Exception as e:
                logger.error(f"Error loading inbound-trunk.json: {e}")
                raise

            # Verify trunk_ids in dispatch-rule.json are not using the placeholder
            if "ST_REPLACE_WITH_LIVEKIT_INBOUND_TRUNK_ID" in dispatch_rule.get("trunk_ids", []):
                raise ValueError("Dispatch rule contains placeholder trunk ID")

            self.test_results.append({"name": "SIP Bridge Configuration", "status": "PASSED"})
            logger.info("SIP Bridge Configuration test PASSED")

        except Exception as e:
            logger.error(f"SIP Bridge Configuration test FAILED: {e}")
            self.test_results.append({"name": "SIP Bridge Configuration", "status": "FAILED", "error": str(e)})

    async def run_all_tests(self) -> None:
        """Run all local tests."""
        logger.info("Starting local tests...")

        await self.test_rag_utils()
        await self.test_latency_metrics()
        await self.test_gpu_scheduler()
        await self.test_sip_bridge()

        logger.info("\n=== Test Results ===")
        for result in self.test_results:
            status = result["status"]
            name = result["name"]
            error = result.get("error", "")

            if status == "PASSED":
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED - {error}")

        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        total = len(self.test_results)
        logger.info(f"\nSummary: {passed}/{total} tests passed")

if __name__ == "__main__":
    test_runner = LocalTestRunner()
    asyncio.run(test_runner.run_all_tests())