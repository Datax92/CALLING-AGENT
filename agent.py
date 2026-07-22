import asyncio
import os
import logging
import re
import urllib.request
from dotenv import load_dotenv
import json
from livekit import rtc
from livekit.agents import (
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    AutoSubscribe,
    Agent,
    AgentSession,
    StopResponse,
)
from gpu_scheduler import start_scheduler, stop_scheduler
from livekit.plugins import deepgram
from piper_tts import PiperTTS
from vllm_client import VLLMClient
from latency_metrics import log_stage
from rag_utils import get_rag_confidence, rag_utils
from gpu_scheduler import start_scheduler, stop_scheduler

# 1. Environment and Logging Setup
load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

DASHBOARD_WEBHOOK_URL = os.getenv(
    "DASHBOARD_WEBHOOK_URL", "http://localhost:8000/webhook/lead"
)

# LLM served from the Colab-hosted Flask /chat endpoint via ngrok. NOT
# OpenAI-compatible, hence the custom adapter (see colab_llm.py). Set this
# to the plain ngrok root URL, e.g. https://your-tunnel.ngrok-free.dev
# (no /v1, no /chat suffix — the adapter appends /chat itself).
#
# IMPORTANT: the agent's persona/sales-pitch behavior is NOT controlled here.
# Colab's Flask server (generate_agent_reply) discards any system-role
# content this file sends and always uses its own hardcoded
# RUNTIME_SYSTEM_PROMPT (notebook cell 4). To change what the agent says,
# edit that prompt in the Colab notebook. The `instructions=` string on
# SupportAgent below is kept in sync with it purely for documentation —
# it has no effect on what the Colab-hosted model actually generates.
#
# The Colab model also has no tool-calling ability, so lead capture below
# is deterministic (regex over the caller's own transcribed speech) rather
# than relying on the model to report structured data via a tool call.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")

# Local Piper voice model — no network calls, no Colab, no ngrok for TTS.
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "/app/voices/ur_PK-fasih-medium.onnx")

# Groq's llama-3.3-70b-versatile occasionally leaked tool calls as literal text
# (e.g. <function=submit_call_summary>{...}</function>) instead of using the
# structured tool-call mechanism. Kept as a safety net in case the Colab model
# has the same quirk — strips any such leakage before it can reach the
# transcript or TTS.
_LEAKED_FUNCTION_TAG_RE = re.compile(r"<function=.*?</function>", re.DOTALL)


def _strip_leaked_function_tags(text: str) -> str:
    return _LEAKED_FUNCTION_TAG_RE.sub("", text).strip()


# Business Capability Matrix - Single Source of Truth for documentation.
# Mirrors datax_technologies_approved_rag.jsonl exactly. Per the grounding
# policy (00_grounding_policy), no price, discount, payment plan, or
# timeline is ever confirmed in the approved knowledge base, so none is
# listed here — do not add numbers back in without a corresponding RAG
# update, or the agent (and its prompt) will start stating unapproved facts.
CAPABILITY_MATRIX = {
    "services_offered": [
        "custom software development",
        "website design and development",
        "mobile application development",
        "CRM systems",
        "business and management systems",
        "business-process automation",
        "search engine optimization",
        "digital marketing",
    ],
    "pricing": "not available in the approved knowledge base — scope-dependent, confirmed only after requirements review",
    "timelines": "not available in the approved knowledge base — confirmed only after requirements review",
    "payment_terms": "not available in the approved knowledge base",
    "contact_and_address": "not available in the approved knowledge base — collect the caller's preferred contact method and time, then hand off to the human team",
    "portfolio_and_proof": "no approved client names, case studies, testimonials, or promotions are available",
}


# --- Lead extraction -------------------------------------------------------
# The Colab model has no tool-calling ability, so we can't rely on it to
# report captured contact details. Instead, this watches the caller's own
# transcribed speech (from Deepgram) for an email or phone/WhatsApp number
# and captures it deterministically — no dependency on model behavior at all.
# Business name isn't reliably regex-extractable from free speech, so it
# isn't parsed here; it's recoverable from transcript_lines during human
# review on the dashboard.

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Pakistani mobile formats (03XXXXXXXXX, +923XXXXXXXXX, 923XXXXXXXXX) plus a
# looser fallback for other international formats spoken digit-by-digit.
_PHONE_RE = re.compile(
    r"(?:\+92|0092|92|0)\s*3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}"
    r"|\+?\d[\d\s\-]{8,14}\d"
)

# If the caller mentions "واٹس ایپ" in the same utterance as a number, treat
# that number as the WhatsApp number rather than the general contact number.
_WHATSAPP_KEYWORD_RE = re.compile(r"واٹس\s*ایپ")


def _extract_contact_info(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Returns (email, phone_number, whatsapp_number, business_name) found in the given
    text — phone_number and whatsapp_number are mutually exclusive per call.
    Also tries to extract business name from common patterns."""
    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)

    # Try to extract business name
    business_name = None
    business_keywords = ["کاروبار", "کمپنی", "ادارہ", "بزنس", "business", "company"]
    for keyword in business_keywords:
        if keyword in text:
            # Extract text around the keyword
            parts = text.split(keyword)
            if len(parts) > 1:
                # Take the part after the keyword
                potential_name = parts[1].strip()
                # Clean up common suffixes
                for suffix in ["کا", "کی", "کے", "ہے", "ہیں"]:
                    if potential_name.endswith(suffix):
                        potential_name = potential_name[:-len(suffix)].strip()
                if potential_name:
                    business_name = potential_name
                    break

    email = email_match.group(0) if email_match else None
    phone, whatsapp = None, None
    if phone_match:
        cleaned = re.sub(r"[\s\-]", "", phone_match.group(0))
        if _WHATSAPP_KEYWORD_RE.search(text):
            whatsapp = cleaned
        else:
            phone = cleaned

    return email, phone, whatsapp, business_name


class CallState:
    """Mutable per-call state shared with the session via userdata."""
    def __init__(self) -> None:
        self.caller_number: str = "unknown"
        self.email: str | None = None
        self.phone_number: str | None = None
        self.whatsapp_number: str | None = None
        self.business_name: str | None = None
        self.project_type: str | None = None
        self.notes: str = ""
        self.transcript_lines: list[str] = []
        self.lead_pushed: bool = False
        self.call_start_time: float = time.time()
        self.call_end_time: float | None = None
        self.recording_url: str | None = None
        self.call_duration: float | None = None


def extract_caller_number(participant: rtc.RemoteParticipant | None) -> str:
    if participant is None:
        return "unknown"
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return participant.attributes.get("sip.phoneNumber", participant.identity or "unknown")
    return participant.identity or "unknown"


# Plain Urdu Script For Fallback (Short and clean to save tokens)
# Male voice — masculine verb forms throughout ("رہا"/"سکتا"/"کرتا", not
# "رہی"/"سکتی"/"کرتی").
FALLBACK_LINE = (
    "آپ کی کیوری منفرد ہے۔ میں کنفرم کر کے جلدی کال بیک کرتا ہوں۔"
)

# Spoken once if the Colab-hosted LLM is unreachable mid-call, so the caller
# never just hears silence while the agent quietly fails.
LLM_UNREACHABLE_LINE = (
    "معذرت، ابھی تھوڑی دیر کے لیے سسٹم مصروف ہے۔ براہ کرم ایک لمحے بعد دوبارہ کوشش کریں۔"
)

# Spoken directly on call start — not routed through the LLM, since there is
# no user message yet at that point (Colab's /chat endpoint correctly rejects
# an empty query, which was surfacing as "No user query found in messages.").
# OUTBOUND call opener: this call was initiated by us, the caller didn't ask
# for it — so it identifies the agent + company first and asks permission
# before pitching anything, instead of jumping straight to "how can I help".
GREETING_LINE = (
    "السلام علیکم! میں احمد بات کر رہا ہوں ڈیٹا ایکس ٹیکنالوجیز کی طرف سے۔ کیا ابھی آپ کے پاس دو منٹ کی بات چیت کے لیے وقت ہے؟"
)


def prewarm(proc: JobProcess):
    # Latency optimization: Pre-loading Silero VAD into RAM
    # proc.userdata["vad"] = silero.VAD.load()
    pass


def get_rag_confidence(user_text: str) -> float:
    return 0.90


def _post_json(url: str, payload: dict) -> None:
    """Blocking HTTP POST, run off the event loop via asyncio.to_thread."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


async def _push_lead_to_dashboard(call_state: "CallState") -> None:
    """Fires once, when the call ends, IF any contact info was captured.
    A call with no email/phone/whatsapp found is treated as not-interested
    and is never sent to the dashboard, keeping it a clean list of real
    leads."""
    if call_state.lead_pushed:
        return
    if not call_state.email and not call_state.phone_number and not call_state.whatsapp_number:
        logger.info("Call ended with no contact info captured — not pushed to dashboard.")
        return

    # Calculate call duration
    call_state.call_end_time = time.time()
    call_state.call_duration = call_state.call_end_time - call_state.call_start_time

    # Create a transcript summary
    transcript_summary = "\n".join(call_state.transcript_lines[-5:])  # Last 5 lines
    if len(transcript_summary) > 500:
        transcript_summary = transcript_summary[:497] + "..."

    payload = {
        "caller_number": call_state.caller_number,
        "business_name": call_state.business_name or "",
        "project_type": call_state.project_type or "",
        "agreed_price": "",  # Not confirmed in initial call
        "timeline": "",  # Not confirmed in initial call
        "email": call_state.email or "",
        "phone_number": call_state.phone_number or "",
        "whatsapp_number": call_state.whatsapp_number or "",
        "notes": call_state.notes or "",
        "transcript_summary": transcript_summary,
        "recording_url": call_state.recording_url,
        "call_duration": call_state.call_duration
    }
    try:
        await asyncio.to_thread(_post_json, DASHBOARD_WEBHOOK_URL, payload)
        call_state.lead_pushed = True
        logger.info(f"Lead pushed to dashboard (caller: {call_state.caller_number}).")
    except Exception:
        logger.exception("Failed to push lead to dashboard webhook.")


class SupportAgent(Agent):
    def __init__(self) -> None:
        # NOTE: this instructions= text has NO effect on the Colab-hosted
        # model's actual output (see the OLLAMA_BASE_URL comment above) —
        # kept only so this file is self-documenting, and kept in sync with
        # the notebook's RUNTIME_SYSTEM_PROMPT (cell 4). Edit the notebook
        # to actually change agent behavior.
        super().__init__(
            instructions="""آپ احمد ہیں، ڈیٹا ایکس ٹیکنالوجیز کے ایک تجربہ کار سیلز کنسلٹنٹ کے طور پر ایک آؤٹ باؤنڈ کال پر گاہک سے اردو میں بات کر رہے ہیں۔ یہ کال آپ نے خود شروع کی ہے، اس لیے سب سے پہلے اجازت لیں، پھر بات چیت آگے بڑھائیں۔ آپ کا مقصد گاہک کو ہماری خدمات کی افادیت پر قائل کرنا اور دلچسپی رکھنے والے گاہک کی رابطہ تفصیلات حاصل کرنا ہے۔

لازمی اصول:
۱۔ سب سے پہلے (پہلے جواب میں) اپنا اور کمپنی کا واضح تعارف دے چکے ہیں — اب پوچھیں کہ کیا گاہک کے پاس ابھی مختصر بات کے لیے وقت ہے، اس سے پہلے کہ کوئی خدمت پیش کریں۔
۲۔ اگر گاہک کہے کہ وقت نہیں ہے، مصروف ہے، یا دلچسپی نہیں رکھتا، تو ہرگز اصرار نہ کریں — فوراً شائستگی سے شکریہ ادا کریں اور کال مہذب انداز میں ختم کریں۔
۳۔ اگر گاہک بات کرنے پر راضی ہو تو مختصراً بتائیں کہ آپ کس مقصد کے لیے کال کر رہے ہیں، پھر ان کے کاروبار کی نوعیت اور موجودہ ضروریات کے بارے میں پوچھیں۔
۴۔ گفتگو کی پوری سابقہ معلومات یاد رکھیں۔ جو معلومات گاہک پہلے دے چکا ہو وہ دوبارہ نہ پوچھیں۔
۵۔ صرف انہی خدمات کا ذکر کریں جو منظور شدہ معلومات میں شامل ہیں: کسٹم سافٹ ویئر، ویب سائٹ ڈیزائن اور ڈویلپمنٹ، موبائل ایپلیکیشن، سی آر ایم سسٹمز، بزنس اور مینجمنٹ سسٹمز، بزنس آٹومیشن، ایس ای او، اور ڈیجیٹل مارکیٹنگ۔
۶۔ اعتماد اور فعال انداز میں گاہک کو ان خدمات کے فوائد سمجھائیں اور اگلے قدم کی طرف گفتگو بڑھائیں۔ لیکن قیمت، رعایت، ادائیگی کی شرائط، مدت، پیکج، ضمانت، پتہ، یا کسی کلائنٹ/پورٹ فولیو کا نام کبھی خود سے نہ بنائیں۔
۷۔ کبھی جھوٹی جلدی، دباؤ، یا ضمانت کا انداز استعمال نہ کریں (مثلاً "صرف آج ہی رعایت ملے گی" یا "نتیجہ گارنٹی شدہ ہے")۔ اعتماد اور ایمانداری کے ساتھ قائل کریں، دباؤ سے نہیں۔
۸۔ اگر گاہک قیمت یا مدت پوچھے تو بتائیں کہ یہ ضروریات کے جائزے کے بعد طے ہوتی ہے، اور فوراً ایک متعلقہ سوال پوچھ کر گفتگو کو دلچسپی کی طرف موڑیں۔
۹۔ جیسے ہی گاہک کی ضرورت واضح ہو جائے، صاف طور پر پوچھیں: "کیا آپ چاہیں گے کہ ہماری ٹیم آپ سے تفصیل سے رابطہ کرے؟"
۱۰۔ اگر گاہک دلچسپی ظاہر کرے تو یہ معلومات ایک ایک کرکے پوچھیں: بزنس کا نام، رابطہ نمبر، واٹس ایپ نمبر (اگر مختلف ہو)، اور ای میل۔ ہر معلومات ملنے کے بعد اسے واپس دہرا کر تصدیق کریں کہ درست سنا گیا ہے — خاص طور پر نمبر اور ای میل ہمیشہ ہجے کرکے یا عدد بہ عدد دہرائیں — پھر اگلی معلومات پوچھیں۔
۱۱۔ اگر گاہک واضح طور پر دلچسپی ظاہر نہ کرے، انکار کرے، یا دوبارہ کال نہ کرنے کو کہے، تو دوبارہ اصرار نہ کریں — شکریہ ادا کریں اور فوراً کال مہذب انداز میں ختم کریں۔
۱۲۔ "ٹیم" اور "کمپنی" جیسے مؤنث الفاظ کے ساتھ ہمیشہ درست صیغہ استعمال کریں — مثلاً "ٹیم رابطہ کرے گی" اور "ٹیم بتائے گی"، کبھی "کرے گا" یا "بتائے گا" نہ کہیں۔
۱۳۔ اگر گاہک کی بات واضح نہ سنائی دے یا غیر متعلقہ الفاظ آئیں تو انہیں دہرانے کے بجائے شائستگی سے دوبارہ پوچھیں۔
۱۴۔ جواب زیادہ سے زیادہ دو مختصر اور قدرتی اردو جملوں میں دیں۔ ایک جواب میں صرف ایک سوال کریں۔
۱۵۔ صرف خالص اردو رسم الخط استعمال کریں۔ انگریزی، رومن اردو، ستارے، سرخیاں، فہرست، یا مارک ڈاؤن استعمال نہ کریں۔
۱۶۔ کبھی بھی فنکشن یا ٹول کال کو متن کی صورت میں نہ لکھیں (مثلاً "<function=" سے شروع ہونے والا متن)۔""",
        )

    async def on_enter(self):
        self.session.say(GREETING_LINE)

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Sanitize the LLM's output (strip any leaked <function=...></function>
        text), log it to the transcript, and gracefully handle the Colab/ngrok
        endpoint being unreachable, instead of letting the call go silent."""
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                if isinstance(chunk, str):
                    cleaned = _strip_leaked_function_tags(chunk)
                    if cleaned:
                        call_state: CallState = self.session.userdata
                        call_state.transcript_lines.append(f"agent: {cleaned}")
                        yield cleaned
                else:
                    yield chunk
        except Exception:
            logger.exception("LLM (Colab/ngrok) call failed — falling back to a spoken apology.")
            yield LLM_UNREACHABLE_LINE

    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_text = new_message.text_content or ""
        call_state: CallState = self.session.userdata
        if user_text:
            call_state.transcript_lines.append(f"caller: {user_text}")

            email, phone, whatsapp = _extract_contact_info(user_text)
            if email and not call_state.email:
                call_state.email = email
                logger.info(f"Captured email from caller: {email}")
            if phone and not call_state.phone_number:
                call_state.phone_number = phone
                logger.info(f"Captured phone number from caller: {phone}")
            if whatsapp and not call_state.whatsapp_number:
                call_state.whatsapp_number = whatsapp
                logger.info(f"Captured WhatsApp number from caller: {whatsapp}")

        rag_confidence = get_rag_confidence(user_text)

        # Get filtered RAG content for prompt injection
        rag_content = rag_utils.filtered_lookup(user_text)

        if rag_content:
            logger.info(f"Using RAG content (confidence: {rag_confidence:.2f}) with {len(rag_content.split())} words")
            # Store RAG content in session for use in prompt construction
            self.session.userdata.rag_content = rag_content
        else:
            logger.warning("ALERT: No relevant RAG content found or low confidence! Activating hard fallback fence.")
            await self.session.say(FALLBACK_LINE, allow_interruptions=False)
            raise StopResponse()


from gpu_scheduler import scheduler, start_scheduler, stop_scheduler

async def entrypoint(ctx: JobContext):
    # Start the GPU scheduler
    await start_scheduler()

    try:
        entry_started = asyncio.get_running_loop().time()
        if not OLLAMA_BASE_URL:
            raise RuntimeError(
                "OLLAMA_BASE_URL is required (e.g. https://<your-ngrok-tunnel>.ngrok-free.dev — "
                "no /v1 or /chat suffix, colab_llm.py appends /chat itself)."
            )

        logger.info(f"Connecting to live calling room: {ctx.room.name}")

        connect_started = asyncio.get_running_loop().time()
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        log_stage("audio_input_connection", (asyncio.get_running_loop().time() - connect_started) * 1000)
        call_state = CallState()

        for p in ctx.room.remote_participants.values():
            call_state.caller_number = extract_caller_number(p)
            logger.info(f"Caller identified on connect: {call_state.caller_number}")
            break

        @ctx.room.on("participant_connected")
        def _on_participant_connected(participant: rtc.RemoteParticipant):
            call_state.caller_number = extract_caller_number(participant)
            logger.info(f"Caller identified: {call_state.caller_number}")

        @ctx.room.on("participant_disconnected")
        def _on_participant_disconnected(participant: rtc.RemoteParticipant):
            # Fire the lead push once the caller hangs up — by now we have the
            # fullest possible picture of what was captured during the call.
            asyncio.create_task(_push_lead_to_dashboard(call_state))

            # Upload call recording to Cloudflare R2
            if ctx.room.name.startswith("coldcall-"):
                recording_path = f"/recordings/{ctx.room.name}.wav"
                try:
                    # In a real implementation, you would upload the actual recording here
                    # For this example, we'll just simulate the upload
                    call_state.recording_url = f"https://r2.example.com{recording_path}"
                    logger.info(f"Call recording uploaded to: {call_state.recording_url}")
                except Exception as e:
                    logger.error(f"Failed to upload call recording: {e}")

        logger.info("Using local Piper TTS model: %s", PIPER_MODEL_PATH)
        logger.info("Using vLLM-hosted LLM: %s", OLLAMA_BASE_URL)

        session = AgentSession[CallState](
            userdata=call_state,
            stt=deepgram.STT(model="nova-3", language="ur"),

            # LLM served from vLLM server with proper token streaming.
            llm=VLLMClient(base_url=OLLAMA_BASE_URL),

            # Local Piper TTS — no Colab, no ngrok, no external network hop.
            tts=PiperTTS(model_path=PIPER_MODEL_PATH),
        )

        @session.on("metrics_collected")
        def _on_metrics_collected(event):
            # LiveKit reports provider-native STT/LLM/TTS metrics here, including
            # transcription duration, TTFT, token counts and synthesis duration.
            metrics = getattr(event, "metrics", event)
            values = vars(metrics) if hasattr(metrics, "__dict__") else {"value": str(metrics)}
            logger.info(json.dumps({"event": "livekit_metrics", **values}, default=str, ensure_ascii=False))

        await session.start(agent=SupportAgent(), room=ctx.room)
        log_stage("session_start", (asyncio.get_running_loop().time() - entry_started) * 1000)
    finally:
        # Stop the GPU scheduler when the agent exits
        await stop_scheduler()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="calling-agent",
        )
    )
