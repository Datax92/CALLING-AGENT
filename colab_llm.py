"""
Custom LiveKit LLM plugin for the Colab-hosted Qwen3.5-4B Flask server.

The Colab notebook (qwen35_4b_unsloth_urdu_qlora_t4_NGROK.ipynb) exposes a
single, non-streaming, non-OpenAI-compatible endpoint:

    POST {base_url}/chat
    body: {"history": [{"role": "user"|"assistant", "content": "..."}]}
    resp: {"reply": "..."}  (or {"error": "..."} with HTTP 500)

This plugin talks to that endpoint directly instead of forcing it through
livekit's openai.LLM client, which expects /v1/chat/completions and a
completely different request/response shape.
"""
import asyncio
import json
import urllib.error
import urllib.request
import logging
import time
import uuid

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.llm import ChatChunk, ChoiceDelta
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from latency_metrics import log_stage

logger = logging.getLogger("voice-agent.llm")


class ColabLLM(llm.LLM):
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        super().__init__()
        # base_url should be the ngrok root, e.g. https://xxxx.ngrok-free.dev
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model(self) -> str:
        return "qwen3.5-4b-urdu-colab"

    @property
    def provider(self) -> str:
        return "colab-flask"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> "ColabLLMStream":
        return ColabLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class ColabLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        turn_id = uuid.uuid4().hex[:12]
        turn_started = time.perf_counter()
        prompt_started = time.perf_counter()
        history = []
        for item in self._chat_ctx.items:
            if getattr(item, "type", None) != "message":
                continue
            if item.role not in ("user", "assistant"):
                continue
            text = item.text_content
            if text:
                history.append({"role": item.role, "content": text})

        log_stage(
            "prompt_construction",
            (time.perf_counter() - prompt_started) * 1000,
            turn_id=turn_id,
            history_messages=len(history),
            prompt_chars=sum(len(x["content"]) for x in history),
        )

        url = f"{self._llm._base_url}/chat"
        body = json.dumps({"history": history, "turn_id": turn_id}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )

        def _do_request() -> dict:
            with urllib.request.urlopen(req, timeout=self._llm._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        http_started = time.perf_counter()
        try:
            result = await asyncio.to_thread(_do_request)
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise APIStatusError(
                message=body_text, status_code=e.code, retryable=(e.code >= 500)
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

        reply = result.get("reply")
        if reply is None:
            raise APIStatusError(
                message=result.get("error", "no 'reply' field in Colab response"),
                status_code=502,
                retryable=True,
            )

        response_ms = (time.perf_counter() - http_started) * 1000
        server_metrics = result.get("metrics", {})
        log_stage(
            "external_llm_api_complete",
            response_ms,
            turn_id=turn_id,
            buffered=True,
            output_chars=len(reply),
            server_metrics=server_metrics,
        )
        # The endpoint is non-streaming: TTFT equals full HTTP response time.
        log_stage("llm_first_token_visible", response_ms, turn_id=turn_id, buffered=True)
        self._event_ch.send_nowait(
            ChatChunk(
                id="colab-0",
                delta=ChoiceDelta(role="assistant", content=reply),
            )
        )
        log_stage("llm_turn_total", (time.perf_counter() - turn_started) * 1000, turn_id=turn_id)
