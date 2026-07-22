"""Local Piper TTS adapter with incremental audio emission and timings."""
import asyncio
import time

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from piper import PiperVoice
from latency_metrics import log_stage


class PiperTTS(tts.TTS):
    def __init__(self, model_path: str, config_path: str | None = None) -> None:
        started = time.perf_counter()
        self._voice = PiperVoice.load(model_path, config_path=config_path)
        log_stage("tts_model_loading", (time.perf_counter() - started) * 1000,
                  model_path=model_path)
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=self._voice.config.sample_rate, num_channels=1)

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class PiperChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        voice: PiperVoice = self._tts._voice  # type: ignore[attr-defined]
        text = self.input_text
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

        def generate() -> None:
            try:
                for chunk in voice.synthesize(text):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.audio_int16_bytes)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        started = time.perf_counter()
        output_emitter.initialize(request_id=str(id(self)), sample_rate=voice.config.sample_rate,
                                  num_channels=1, mime_type="audio/pcm")
        producer = asyncio.create_task(asyncio.to_thread(generate))
        chunks = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise tts.APIConnectionError() from item
            if chunks == 0:
                log_stage("tts_first_audio_ready", (time.perf_counter() - started) * 1000,
                          input_chars=len(text), buffered=False)
            output_emitter.push(item)
            chunks += 1
        await producer
        output_emitter.flush()
        log_stage("tts_complete", (time.perf_counter() - started) * 1000,
                  input_chars=len(text), chunks=chunks, buffered=False)
