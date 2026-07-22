# Urdu voice-agent latency analysis

## Actual request flow

Caller audio -> Twilio/SIP -> LiveKit audio track -> Deepgram Nova-3 Urdu STT
over the network -> LiveKit conversation history -> `ColabLLM` -> HTTPS/ngrok ->
Flask `/chat` -> tokenizer/chat template -> Qwen3.5-4B + QLoRA, bitsandbytes
4-bit, Transformers/Unsloth on Colab CUDA -> output validation (up to three full
generations) -> buffered JSON/ngrok response -> regex post-processing -> local
Piper `ur_PK-fasih-medium` CPU synthesis -> LiveKit audio -> caller.

Lead extraction is local regex and the dashboard POST happens after disconnect,
so neither is on conversational response latency. There is no implemented RAG
pipeline: embedding, FAISS search, chunk retrieval and correction API are all
absent and therefore currently cost 0 ms.

## Evidence-backed findings

1. **Critical – LLM is fully buffered.** `model.generate()` completes before
   Flask returns JSON; `ColabLLMStream` then emits the entire reply as one chunk.
   Client-observed TTFT equals complete inference plus two network/tunnel legs.
2. **Critical – validation may generate 2–3 complete answers.** The sampled pass
   is followed by a greedy pass on any validation failure, then a 280-token pass
   when truncation is the only failure. Worst case is 680 generated tokens.
3. **High – TTS was fully buffered.** The adapter collected every Piper chunk in
   a list before emitting audio. This has been changed to emit each chunk as it
   arrives, preserving the same voice and text.
4. **High – unbounded conversation history.** Every user/assistant message is
   sent each turn. Prompt evaluation therefore grows throughout a call and can
   eventually collide with the 1,536-token context.
5. **High – remote Colab/ngrok hop.** STT already makes a network call, then LLM
   adds a separate public tunnel round trip. Colab sleep/restart also adds cold
   starts outside the request handler.
6. **Medium – 200 `max_new_tokens` is high for a two-sentence voice reply.** A
   truncation retry raises it to 280. Start at 120 after measuring Urdu outputs.
7. **Medium – per-token full-vocabulary mask.** `UrduOnlyLogitsProcessor` runs a
   `masked_fill` across the complete vocabulary every token. Preserve it for
   Urdu correctness until profiling proves its cost; a compiled/static mask or
   constrained vocabulary backend can reduce it.
8. **Low – production model and Piper objects are reused correctly.** Qwen is
   loaded once in notebook cell 4. Piper is created once per LiveKit job/call,
   not per utterance; worker-process prewarming could also remove call-start
   loading.
9. **Low – Gradio debug/share mode is enabled.** This creates an unnecessary UI,
   public tunnel and logging in the same Colab runtime.

## Backend verification

The active LLM is not llama.cpp, llama-cpp-python, Ollama or GGUF, so
`n_gpu_layers`, llama.cpp threads/batch size, and CUDA-wheel verification do not
apply. The notebook requires CUDA and uses Unsloth/Transformers with
bitsandbytes 4-bit Qwen3.5-4B, `MAX_SEQ_LENGTH=1536`, FP16 or BF16 compute, and
KV cache. No explicit inference batch size or CPU thread count is configured;
single-request batch size is effectively one. Model artifact byte size is not
present in this repository.

Successful package installation is not treated as CUDA proof. The server patch
logs both `torch.cuda.is_available()` and `next(model.parameters()).is_cuda`,
GPU name and VRAM. For any future llama-cpp deployment, verify at runtime with
`llama_cpp.llama_supports_gpu_offload()` plus verbose model-load logs showing
layers offloaded; this repository cannot truthfully run that check because it
does not depend on llama-cpp-python.

Local inspection found Python 3.14 and no NVIDIA runtime, but production's
Dockerfile pins Python 3.11. Therefore Python 3.14 is not a production bottleneck.

## Measurement status and instrumentation

No valid current milliseconds can be measured in this checkout: `.env.local`,
LiveKit, Deepgram, remote Colab endpoint and local Piper model are unavailable.
Invented numbers would not identify a bottleneck. Added JSON logs cover audio
connection, LiveKit-native STT/LLM/TTS metrics, prompt construction, external
LLM HTTP total, visible TTFT, complete LLM turn, TTS model load, TTS first audio,
TTS complete, CPU/RAM/GPU/VRAM, history size and prompt characters. Apply
`COLAB_LATENCY_PATCH.md` to add prompt/output token counts, tokens/sec, CUDA,
VRAM, generation attempts and synchronized server generation time.

Run a real call and filter logs for `"event": "latency"` and
`"event": "livekit_metrics"`. Use at least 10 warm turns and report p50/p95;
separate the first cold call. The existing benchmark cannot measure production
today because it waits for nonexistent `benchmark` data messages and labels an
LLM phase `mock_response_generated`.

## Fix order and expected improvement

Expected gains are ranges, not measured claims; the new baseline logs will
replace them with exact values.

1. Stream LLM tokens through Flask and `ColabLLMStream`, then sentence-chunk
   into TTS: **TTFT improvement 60–90%**, perceived response start commonly
   several seconds earlier.
2. Avoid unconditional regeneration; validate incrementally or repair only the
   violated suffix: **0–67% generation-time reduction**, depending on retry rate.
3. Keep a bounded history (for example last 6 messages plus a compact state
   summary): **10–50% prompt-evaluation reduction** late in calls.
4. Use 120 output tokens initially and stop on the first complete sentence pair:
   **20–40% generation reduction** on previously long/truncated turns.
5. Incremental Piper emission (implemented): **50–90% lower TTS time-to-first-
   audio**, with similar total synthesis time.
6. Prewarm Piper per worker rather than per call: removes the logged model-load
   time from call startup; normally **hundreds of ms to seconds once per call**.
7. Replace Colab/ngrok with a persistent nearby GPU service using vLLM/SGLang:
   removes tunnel variance/cold starts and enables native streaming; improvement
   depends on geography and Colab state.

## Concise component table

| Component | Current latency | Detected problem | Recommended fix | Expected after optimization |
|---|---:|---|---|---:|
| Audio/turn detection | Not measurable here | External LiveKit path | Use emitted LiveKit STT/EOU metrics | Baseline-dependent |
| Deepgram STT | Not measurable here | External network service | Measure p50/p95; keep streaming connection | Baseline-dependent |
| Embedding/FAISS/retrieval | 0 ms | Not implemented | No change | 0 ms |
| Prompt construction/eval | Not measurable here | History grows unbounded | Last 6 messages + state summary | 10–50% lower late-call |
| LLM/network TTFT | Equals full HTTP completion | Entire response buffered | Streaming server/client | 60–90% lower TTFT |
| LLM generation | Not measurable here | Up to 3 runs / 680 tokens | One pass, 120-token cap | 20–67% lower |
| Post-processing | Near-trivial, unmeasured | Regex only | Keep | Essentially unchanged |
| Piper model load | Not measurable here | Once per call | Worker prewarm/cache | Removed from call path |
| Piper first audio | Previously full synthesis | Full list buffering | Incremental chunks (done) | 50–90% lower |
| Dashboard webhook | 0 ms in response path | Runs after hangup | Keep asynchronous | 0 ms |
