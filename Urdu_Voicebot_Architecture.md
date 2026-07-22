# Urdu Voicebot — Complete Architecture Reference

**System:** Inbound/outbound Urdu voice agent (cold-calling / customer support)
**Model:** Qwen3.5-4B, 4-bit QLoRA fine-tune, Urdu
**Status:** Production architecture proposal — supersedes the Colab/ngrok prototype
**Last updated:** July 2026

---

## 1. Purpose & Scope

This document is the single technical reference for the voicebot's production deployment: every
component, how they connect, why each choice was made, what's already built vs. what still needs
to change, and every constraint or open risk that affects reliability, latency, or cost. It's
written for engineers picking up the build, not just for budget sign-off.

**Design targets:**
- End-to-end turn latency (caller stops speaking → bot starts speaking): **p95 ≤ 650ms**
- Language: Urdu only, streaming STT and local TTS
- Call volume basis: ~5,000 active call minutes/month
- No paid vector database, no idle 24/7 GPU spend, no cold starts on live calls

---

## 2. High-Level Architecture

```text
                                +--------------------------+
                                |     PSTN / Phone Net     |
                                |   (Pakistani Mobiles)    |
                                +-------------+------------+
                                              |
                                +-------------v------------+
                                |        PTCL SIP           |
                                |  (Business SIP Trunk)     |
                                +-------------+------------+
                                              | SIP over TLS/SRTP
                                +-------------v------------+
        +---------------------+|      LiveKit Cloud        |+---------------------+
        |                      ||  (SIP Trunk + WebRTC SFU) ||                      |
        |                      |+-------------+------------+|                      |
        | WebRTC Audio                        | WebRTC Audio                       | Egress WAV/MP3
        |                                     |                                    |
+-------v--------+           +----------------v-----------------+          +-------v--------+
|   Deepgram      |           |     CONSOLIDATED APP SERVER      |          | Cloudflare R2   |
|   Nova-3 Urdu   |==========>|      (Railway, compute-tier)     |          | (Call Recording |
|   (Streaming    |  WebSocket|  - Voice Agent Loop (LiveKit SDK)|          |   Archive)      |
|    STT)         |           |  - FastAPI Dashboard             |          +-------+---------+
+-----------------+           |  - Piper TTS (local, CPU-bound,  |                  ^
                               |    sized for concurrent calls)   |                  | URL reference
                               |  - RAG: filtered JSON lookup,    |                  |
                               |    <500 tok injected per turn    |                  |
                               +----------------+------------------+<-----------------+
                                                |         |
                        Prompt + filtered RAG   |         | Call metadata / leads
                        chunk (HTTPS, streamed) |         v
                                                |   +-------------------+
                                                |   |  MongoDB Atlas    |
                                                |   |  (Shared/M0-M10)  |
                                                |   +-------------------+
                                                v
                      +-----------------------------------------------------+
                      |         GPU INFERENCE LAYER (dedicated, split)      |
                      |         no scale-to-zero — no cold starts           |
                      |                                                     |
                      |  Peak hours (~12h/day)     Off-peak (~12h/day)      |
                      |  +-------------------+     +---------------------+  |
                      |  | Dedicated Pod A   |     | Dedicated Pod B     |  |
                      |  | RunPod Community  |     | RunPod Community    |  |
                      |  | A5000/T4 class    |     | RTX 3060 class      |  |
                      |  | vLLM / SGLang     |     | (lighter, cheaper   |  |
                      |  | serving:          |     | tier — 4B QLoRA     |  |
                      |  | Qwen3.5-4B+QLoRA  |     | fits comfortably)   |  |
                      |  | 4-bit, streaming  |     | vLLM / SGLang       |  |
                      |  | tokens            |     | Started right after |  |
                      |  | Stopped by        |     | Pod A stops —       |  |
                      |  | scheduled job     |     | sequential, shares  |  |
                      |  +-------------------+     | same network volume |  |
                      |                            +---------------------+  |
                      +-----------------------------------------------------+
```

---

## 3. Call Sequence (Inbound Example)

Full request path, matching what `sip_bridge.py`, `dispatch-rule.json`, `colab_llm.py`, and
`piper_tts.py` already implement (or need to migrate to):

1. Caller dials the PTCL number → **PTCL SIP trunk** delivers the call over SIP/TLS.
2. **Twilio webhook path is not used for PTCL** — PTCL is a direct SIP interconnect, not a Twilio
   number. (If a Twilio-fronted flow is kept for a *different* number, `sip_bridge.py`'s
   `_twilio_signature_is_valid` HMAC check and `is_allowed_source` caller-ID filter still apply
   there — see §7.4.)
3. LiveKit's **SIP inbound trunk** (`inbound-trunk.json`) receives the INVITE and, per the
   **dispatch rule** (`dispatch-rule.json`), creates a new room named `coldcall-<slug>` and
   auto-dispatches the `calling-agent` agent into it.
4. The **Voice Agent Loop** (LiveKit Agents SDK, running on the app server) joins the room,
   subscribes to the caller's audio track, and streams it to **Deepgram Nova-3 (Urdu)** over
   WebSocket for real-time transcription.
5. As Deepgram returns transcribed text, it's appended to the **LiveKit ChatContext** (conversation
   history).
6. The agent's `ColabLLM`/production LLM client (`colab_llm.py` → to be replaced by a vLLM/SGLang
   client) builds the prompt: system prompt + **filtered RAG chunk** (relevant policy snippet,
   <500 tokens, not the full 20KB file) + bounded conversation history + latest user turn.
7. The prompt is sent to whichever **GPU pod is currently active** (Pod A during peak hours, Pod B
   off-peak) via HTTP, and tokens stream back as they're generated.
8. Generated text is sentence-chunked and sent to **Piper TTS**, which synthesizes and streams
   audio incrementally (`piper_tts.py`'s `PiperChunkedStream` — chunk-by-chunk, not buffered).
9. Synthesized audio is published back into the LiveKit room; the caller hears the reply.
10. Every stage above is timestamped via `latency_metrics.log_stage()` and emitted as a structured
    `{"event": "latency", ...}` JSON log line for the latency dashboard.
11. On call end, lead/summary data is extracted with local regex (no LLM call needed) and POSTed to
    the **FastAPI dashboard's** `/webhook/call-summary` endpoint (see `docker-compose.yml`), which
    writes to **MongoDB Atlas**.
12. Full call audio is archived to **Cloudflare R2**; the dashboard stores/serves the R2 URL, not
    the audio itself.

**Outbound calls** follow the same LLM/STT/TTS path but originate from `sip_bridge.py`'s
`/api/outbound/call` → Twilio REST API (if outbound uses Twilio) or a PTCL-side origination trunk
→ LiveKit outbound SIP participant, joining a fresh `coldcall-` room the same way.

---

## 4. Component Reference

### 4.1 Telephony — PTCL SIP Trunk

| Property | Detail |
|---|---|
| Role | Inbound/outbound PSTN ingress for Pakistani (+92) numbers |
| Protocol | SIP over TLS, media over SRTP |
| Pricing | **Not public — quote-based.** Estimated ~$35/month pending formal PTCL quote |
| Alternative comparison | DIDWW, DIDLogic, or Nayatel (specialized DID/SIP providers) |
| Explicitly ruled out | **Twilio** — confirmed via Twilio's own Pakistan pricing page: no local +92 numbers available, and PSTN termination into Pakistan runs $0.155/min (landline) to $0.18/min (mobile), which is not a viable trunk substitute |
| Constraint | Business SIP trunk contracts in Pakistan typically require in-country registration/paperwork; lead time should be budgeted separately from technical build time |

### 4.2 Voice Orchestration — LiveKit Cloud

| Property | Detail |
|---|---|
| Role | WebRTC SFU + SIP bridging + agent dispatch + room lifecycle |
| Plan | Ship ($50/month) — 5,000 agent-session minutes + 150,000 WebRTC minutes included |
| Config files already present | `dispatch-rule.json` (routes inbound SIP → `coldcall-` prefixed rooms, auto-dispatches `calling-agent`), `inbound-trunk.json` (Twilio-fronted inbound trunk config, `krisp_enabled: true` for noise suppression) |
| **Open risk** | Telephony/PSTN-leg minutes may be metered **separately** from the 5,000 included agent-session minutes (LiveKit's own published example prices telephony at ~$0.01/min on top). Not yet confirmed — budgeted as a $50/month contingency until verified against the live usage dashboard |
| HIPAA/compliance | Not applicable at Ship tier; would require Scale tier ($500/mo) if ever needed |

### 4.3 Speech-to-Text — Deepgram Nova-3

| Property | Detail |
|---|---|
| Role | Real-time streaming Urdu transcription |
| Mode | Streaming (not batch) — required for live-call latency |
| Pricing | **$0.0077/min**, verified against Deepgram's current published rate. At 5,000 min/month: **$38.50/month** |
| Model variant | Monolingual (Urdu) — cheaper than multilingual ($0.0092/min) and correct since audio is single-language |
| Growth-tier note | At >$4,000/year spend, Deepgram Growth pricing gives ~15% discount ($0.0065/min); not relevant at current volume |

### 4.4 App Server — Railway (Consolidated)

Single Railway service running three processes together:

**a) Voice Agent Loop** (LiveKit Agents SDK)
- `latency_metrics.py` — structured JSON latency logging (`log_stage`, `timed` context manager),
  captures CPU/RAM/GPU/VRAM snapshots per stage via `psutil` + `nvidia-smi`
- `colab_llm.py` — current LLM client; **must be replaced** with a vLLM/SGLang-compatible
  streaming client before production (see §6)

**b) FastAPI Dashboard**
- Receives call-summary webhooks, serves the lead/call dashboard UI
- Talks to MongoDB Atlas for storage

**c) Piper TTS** (`piper_tts.py`)
- Local, CPU-based synthesis using `ur_PK-fasih-medium` voice
- Already emits audio **incrementally** (`PiperChunkedStream`) rather than buffering the full
  utterance before playback — this was a completed latency fix, not a pending one
- **Constraint: CPU-bound and shared with the dashboard/agent loop process.** 2–3 concurrent calls
  can saturate CPU, causing stuttering audio or dashboard slowdowns. Requires a compute-optimized
  Railway tier (4+ vCPUs) and active CPU monitoring, not just RAM sizing

**d) RAG (in-process, no vector DB)**
- Knowledge base is a single JSON file, <20KB
- Correct implementation: **filtered lookup**, retrieving only the relevant policy chunk
  (<500 tokens) per turn — not full-file injection. Full-file injection would add
  ~4,000–5,000 tokens to every turn's prompt, which meaningfully inflates time-to-first-token on
  a budget GPU. This must be verified in the actual code before go-live, not assumed

| Property | Detail |
|---|---|
| Plan | Railway Pro, ~$30–40/month (compute-optimized tier to absorb Piper TTS CPU load) |
| Removed from this service | MongoDB (moved to Atlas — see §4.5) |

### 4.5 Database — MongoDB Atlas

| Property | Detail |
|---|---|
| Role | Call metadata, extracted leads, dashboard data |
| Tier | Free (M0) or low-tier shared cluster (M10) — call volume here is metadata-only, not audio |
| Cost | $0–$9/month |
| Why moved off the app server | Removes RAM/disk pressure from the Railway container (which also needs headroom for concurrent Piper synthesis); gains managed backups for free |

### 4.6 Call Recording Archive — Cloudflare R2

| Property | Detail |
|---|---|
| Role | Full-call audio storage (WAV/MP3), referenced by URL from the dashboard |
| Cost | $0.50–$1/month at ~30GB/month (free tier covers most of it, no egress fees) |
| Access pattern | Dashboard stores/displays R2 URLs; playback is direct from R2, not proxied through the app server |

### 4.7 GPU Inference Layer — Split Dedicated Pods

This is the core cost/latency tradeoff in the whole system and the part most likely to need
revision after real usage data comes in.

**Why not 24/7:** A voicebot at 5,000 min/month is not continuously busy; renting GPU time for
hours nobody is calling is pure waste.

**Why not scale-to-zero serverless:** Waking a cold serverless worker means loading ~2.5GB of
model weights into GPU VRAM and initializing vLLM — realistically **8–15+ seconds**. A caller will
not sit through that in silence; they'll assume the call dropped. This was the original design
and was corrected after review.

**Chosen design — two sequential dedicated pods, both always-warm while active:**

| | Pod A (Peak) | Pod B (Off-peak) |
|---|---|---|
| Window | ~12h/day | ~12h/day |
| GPU class | A5000 / T4 | RTX 3060 |
| Rate (RunPod Community Cloud) | ~$0.27–$0.40/hr | ~$0.15–$0.25/hr |
| Monthly (360 hrs) | $97–$144 | $54–$90 |
| Serving stack | vLLM / SGLang | vLLM / SGLang |
| Model | Qwen3.5-4B + QLoRA, 4-bit | same |
| Lifecycle | Started/stopped by scheduled job | Starts right after Pod A stops (sequential, never simultaneous) |
| Storage | Shared RunPod Network Volume (~$3–5/mo) — safe to share because the two pods never run concurrently |

**Automation requirement:** start/stop must be driven by a scheduler (cron or RunPod API trigger),
not manual toggling — a missed stop even a few nights a week erodes most of the savings.

**Cold-start mitigation for the boundary itself:** the handoff between Pod A stopping and Pod B
starting should be scheduled with a short overlap or a prewarm trigger a few minutes before the
switch, so no call lands in the gap.

**Model serving constraint:** `MAX_SEQ_LENGTH=1536` (from the current notebook). Unbounded
conversation history sent every turn will eventually collide with this limit on long calls —
history must be bounded (e.g., last 6 messages + a compact state summary), not sent in full.

---

## 5. Known Constraints From the Current Codebase

These come directly from the uploaded prototype files and apply regardless of which cloud
deployment is chosen:

1. **The current LLM endpoint is fully buffered, non-streaming Flask**, hosted in a Colab notebook
   behind ngrok (`colab_llm.py`, `COLAB_LATENCY_PATCH.md`). `model.generate()` completes before
   Flask returns JSON, and `ColabLLMStream` emits the entire reply as a single chunk. Client-observed
   time-to-first-token equals *complete* inference time plus two network/tunnel hops. **This must be
   replaced by vLLM or SGLang with real token streaming before production** — it's a prerequisite
   for both the latency target and the GPU-hour cost assumptions above, not an optional polish step.
2. **Validation may trigger 2–3 full regenerations per turn.** A sampled pass is followed by a
   greedy retry on validation failure, then up to a 280-token pass if truncation is the only
   failure — worst case 680 generated tokens for one reply. This should be replaced with
   incremental validation or suffix-only repair.
3. **`max_new_tokens` defaults to 200, rising to 280 on retry.** This is high for a two-sentence
   voice reply; start at 120 and stop on the first complete sentence pair.
4. **`UrduOnlyLogitsProcessor` runs a full-vocabulary `masked_fill` every token.** Necessary for
   Urdu correctness — keep it — but it's a per-token cost worth profiling once the rest of the
   pipeline is optimized.
5. **No RAG pipeline currently exists** (no embeddings, no FAISS, no retrieval code) — the <20KB
   JSON knowledge base is small enough that this is correct as designed, provided the filtered
   lookup (§4.4d) is actually implemented rather than full-file injection.
6. **`sip_bridge.py` validates Twilio webhooks via HMAC-SHA1** (`_twilio_signature_is_valid`) and
   filters caller ID against a single allow-listed source number (`is_allowed_source`). This logic
   is specific to a Twilio-fronted number, if one is kept alongside PTCL — it does not apply to
   direct PTCL SIP interconnect, which has its own carrier-level authentication.
7. **Piper TTS and the model objects are already reused correctly** — Piper is instantiated once
   per LiveKit job/call (not per utterance), and the Qwen model is loaded once at notebook/service
   startup. Worker-process prewarming could still remove call-start loading entirely.
8. **Gradio `debug=True, share=True` must be removed** before production — it creates an
   unnecessary public tunnel and UI inside the same runtime as the model server.

---

## 6. Pre-Production Migration Checklist

In priority order, matching the latency-analysis fix order:

- [ ] Port the QLoRA model from the Colab/Transformers/Unsloth prototype to **vLLM or SGLang** with
      native token streaming (expected TTFT improvement: 60–90%)
- [ ] Replace unconditional 2–3x regeneration with incremental validation or suffix repair
      (expected generation-time reduction: 0–67%, retry-rate dependent)
- [ ] Bound conversation history to last ~6 messages + compact state summary instead of full
      unbounded history (expected prompt-eval reduction: 10–50% late in calls)
- [ ] Set `max_new_tokens=120` as the starting point, stop on first complete sentence pair
      (expected generation reduction: 20–40% on previously long/truncated turns)
- [x] Incremental Piper TTS emission — **already implemented** (`piper_tts.py`)
- [ ] Prewarm Piper per worker rather than per call (removes model-load time from call startup)
- [ ] Move off Colab/ngrok to the RunPod dual-pod deployment described in §4.7
- [ ] Confirm RAG implementation does filtered lookup, not full-file prompt injection
- [ ] Remove `demo.launch(share=True, debug=True)` from any Gradio debug UI
- [ ] Automate GPU pod start/stop scheduling (cron or RunPod API) — do not rely on manual ops

---

## 7. Open Items Pending Vendor/Ops Confirmation

| Item | Why it's open | Action needed |
|---|---|---|
| PTCL SIP trunk pricing | Not publicly published, quote-based | Request formal PTCL business quote; get a comparison quote from DIDWW/DIDLogic/Nayatel |
| LiveKit telephony billing | Unclear whether PSTN legs meter separately from agent-session minutes on Ship plan | Verify against live LiveKit usage dashboard once in production; $50/month contingency budgeted until confirmed |
| Off-peak call volume shape | Current 12h/12h GPU split is an assumption, not measured | Collect 2–4 weeks of real call-time logs, then right-size both pod windows |
| Piper concurrency ceiling | Exact number of simultaneous calls before CPU saturation is unmeasured | Load-test on the target Railway compute tier before go-live |

---

## 8. Monthly Cost Summary

*(5,000 active call minutes/month basis; see §4 for per-component detail)*

| Component | Monthly Cost (USD) |
|---|---|
| Core app server (Railway) | $30 – $40 |
| Database (MongoDB Atlas) | $0 – $9 |
| LLM inference — peak pod | $97 – $144 |
| LLM inference — off-peak pod | $54 – $90 |
| Model storage (network volume) | $3 – $5 |
| Voice orchestration (LiveKit Ship) | $50 |
| Speech-to-text (Deepgram Nova-3) | $38.50 |
| Text-to-speech (Piper, local) | $0 |
| Call recording storage (Cloudflare R2) | $0.50 – $1 |
| Telephony (PTCL SIP, estimate) | $35 |
| **Total** | **$308.00 – $412.50** |
| + LiveKit telephony contingency | +$0 – $50 |
| **Total with contingency** | **$308.00 – $462.50** |

For reference, an always-on 24/7 single-GPU version of this same stack costs approximately
$293.50–$434.00/month — the split-pod schedule lands in a similar range while removing cold-start
risk entirely, and leaves room to shrink further once real call-time distribution data narrows the
off-peak pod's hours.

---

## 9. Glossary / File Map

Quick reference from uploaded prototype files to the architecture sections above:

| File | Maps to |
|---|---|
| `agent.py`, `app.py` | Voice Agent Loop / FastAPI Dashboard (§4.4a, §4.4b) |
| `colab_llm.py` | Current (prototype) LLM client — to be replaced (§6) |
| `COLAB_LATENCY_PATCH.md` | CUDA-synchronized generation timing patch for the notebook |
| `piper_tts.py` | TTS adapter (§4.4c) |
| `latency_metrics.py` | Structured latency logging used across all components |
| `benchmark_turn_latency.py` | LiveKit-based E2E turn-latency benchmark harness |
| `LATENCY_ANALYSIS.md` | Source of the fix-order/expected-improvement figures in §6 |
| `sip_bridge.py` | Twilio-fronted SIP bridge logic, if kept alongside PTCL (§5.6) |
| `dispatch-rule.json`, `inbound-trunk.json` | LiveKit SIP dispatch/trunk config (§4.2) |
| `docker-compose.yml`, `Dockerfile.dashboard`, `Dockerfile.voice` | Current local/consolidated deployment definition |
| `render.yaml` | Alternative Render.com deployment definition (not the primary target here) |
| `requirements-dashboard.txt`, `requirements-voice.txt` | Dependency pins for each service |
