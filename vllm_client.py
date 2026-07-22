import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.llm import ChatChunk, ChoiceDelta
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from latency_metrics import log_stage

logger = logging.getLogger("voice-agent.llm")

# Helper to truncate text to a maximum token (word) count. Simple word‑based approximation.
def _truncate_to_tokens(text: str, max_tokens: int = 500) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])

class VLLMClient(llm.LLM):
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Optional model name override via env var
        self._model = os.getenv("VLLM_MODEL", "qwen3.5-4b-urdu-vllm")

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "vllm"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: List[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "VLLMStream":
        return VLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )

class VLLMStream(llm.LLMStream):
    def __init__(
        self,
        client: VLLMClient,
        *,
        chat_ctx: llm.ChatContext,
        tools: List[llm.Tool],
        conn_options: APIConnectOptions,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(client, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs or {}

    async def _run(self) -> None:
        turn_id = uuid.uuid4().hex[:12]
        turn_started = time.perf_counter()
        prompt_started = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Build conversation history – bound to last 6 messages for token safety.
        # ------------------------------------------------------------------
        history: List[Dict[str, str]] = []
        # Reverse iterate to pick most recent messages.
        for item in reversed(self._chat_ctx.items):
            if getattr(item, "type", None) != "message":
                continue
            if item.role not in ("user", "assistant"):
                continue
            if not getattr(item, "text_content", None):
                continue
            history.append({"role": item.role, "content": item.text_content})
            if len(history) >= 6:
                break
        # Reverse back to chronological order.
        history = list(reversed(history))

        # ------------------------------------------------------------------
        # 2. Log prompt construction metrics.
        # ------------------------------------------------------------------
        log_stage(
            "prompt_construction",
            (time.perf_counter() - prompt_started) * 1000,
            turn_id=turn_id,
            history_messages=len(history),
            prompt_chars=sum(len(x["content"]) for x in history),
        )

        # ------------------------------------------------------------------
        # 3. Retrieve optional filtered RAG content from userdata.
        # ------------------------------------------------------------------
        rag_content = ""
        for item in self._chat_ctx.items:
            if hasattr(item, "userdata") and hasattr(item.userdata, "rag_content"):
                rag_content = item.userdata.rag_content or ""
                break
        if rag_content:
            rag_content = _truncate_to_tokens(rag_content, max_tokens=500)

        # ------------------------------------------------------------------
        # 4. Assemble the final prompt for the vLLM server.
        # ------------------------------------------------------------------
        prompt_parts: List[str] = []
        if rag_content:
            prompt_parts.append(f"Relevant Knowledge Base Information:\n{rag_content}\n\n")
        for msg in history:
            prompt_parts.append(f"{msg['role'].upper()}: {msg['content']}\n")
        prompt_parts.append("ASSISTANT: ")
        prompt = "".join(prompt_parts)

        # ------------------------------------------------------------------
        # 5. Send request to vLLM server with streaming response.
        # ------------------------------------------------------------------
        request_payload = {
            "model": self._client.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in history],
            "stream": True,
        }
        # Include RAG content as a system message if present.
        if rag_content:
            request_payload["messages"].insert(0, {"role": "system", "content": rag_content})
        # Forward any extra kwargs (e.g., temperature, max_new_tokens).
        request_payload.update(self._extra_kwargs)

        async with httpx.AsyncClient(timeout=self._client._timeout) as client:
            try:
                async with client.stream("POST", f"{self._client._base_url}/v1/chat/completions", json=request_payload) as response:
                    if response.status_code != 200:
                        raise APIStatusError(
                            f"vLLM request failed with status {response.status_code}",
                            status_code=response.status_code,
                            retryable=response.status_code >= 500,
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content")
                            if content:
                                self._event_ch.send_nowait(
                                    ChatChunk(
                                        id=choice.get("index", 0),
                                        delta=ChoiceDelta(role="assistant", content=content),
                                    )
                                )
            except httpx.RequestError as exc:
                raise APIConnectionError(str(exc)) from exc

        # ------------------------------------------------------------------
        # 6. Log latency metrics after the stream ends.
        # ------------------------------------------------------------------
        response_ms = (time.perf_counter() - turn_started) * 1000
        log_stage(
            "external_llm_api_complete",
            response_ms,
            turn_id=turn_id,
            buffered=False,
            output_chars=None,
        )
        log_stage("llm_first_token_visible", response_ms, turn_id=turn_id, buffered=False)
        log_stage("llm_turn_total", response_ms, turn_id=turn_id)
