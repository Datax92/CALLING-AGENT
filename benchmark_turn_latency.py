import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from dotenv import load_dotenv
from livekit import api, rtc


@dataclass
class RunMetrics:
    voice_input_sent: float | None = None
    text_transcribed: float | None = None
    mock_response_generated: float | None = None
    first_audio_byte_received: float | None = None

    def turnaround_ms(self) -> float | None:
        if self.voice_input_sent is None or self.first_audio_byte_received is None:
            return None
        return (self.first_audio_byte_received - self.voice_input_sent) * 1000.0


def percentile(sorted_values: list[float], p: int) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((p / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def build_token(room_name: str, identity: str) -> str:
    key = os.getenv("LIVEKIT_API_KEY", "")
    secret = os.getenv("LIVEKIT_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required")

    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    return api.AccessToken(key, secret).with_identity(identity).with_grants(grants).to_jwt()


async def send_pcm_5s(track_source: rtc.AudioSource, pcm_path: Path, sample_rate: int = 16000) -> float:
    raw = pcm_path.read_bytes()
    frame_ms = 20
    bytes_per_sample = 2
    frame_samples = int(sample_rate * (frame_ms / 1000.0))
    frame_bytes = frame_samples * bytes_per_sample

    start_ts = asyncio.get_running_loop().time()

    for i in range(0, len(raw), frame_bytes):
        chunk = raw[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            break
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=frame_samples,
        )
        await track_source.capture_frame(frame)
        await asyncio.sleep(frame_ms / 1000.0)

    return start_ts


async def run_once(livekit_url: str, pcm_path: Path, run_id: int) -> RunMetrics:
    room_name = f"bench-{uuid.uuid4().hex[:8]}"
    identity = f"bench-client-{run_id}"
    token = build_token(room_name, identity)

    room = rtc.Room()
    metrics = RunMetrics()
    done = asyncio.Event()

    @room.on("data_received")
    def _on_data(packet: rtc.DataPacket):
        if packet.topic != "benchmark":
            return
        try:
            data = json.loads(packet.data.decode("utf-8"))
        except Exception:
            return
        if data.get("type") != "benchmark":
            return

        now = asyncio.get_running_loop().time()
        stage = data.get("stage")
        if stage == "stt_transcribed" and metrics.text_transcribed is None:
            metrics.text_transcribed = now
        elif stage == "mock_response_generated" and metrics.mock_response_generated is None:
            metrics.mock_response_generated = now
        elif stage == "tts_first_audio_byte" and metrics.first_audio_byte_received is None:
            metrics.first_audio_byte_received = now
            done.set()

    await room.connect(livekit_url, token)

    source = rtc.AudioSource(16000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("bench-input", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    metrics.voice_input_sent = await send_pcm_5s(source, pcm_path)

    try:
        await asyncio.wait_for(done.wait(), timeout=12)
    except asyncio.TimeoutError:
        pass
    finally:
        await room.disconnect()

    return metrics


def print_report(results: list[RunMetrics]) -> None:
    e2e = [x.turnaround_ms() for x in results if x.turnaround_ms() is not None]
    e2e_vals = sorted([x for x in e2e if x is not None])

    print("\n=== Voice Turn Latency Report (10 runs) ===")
    for i, r in enumerate(results, start=1):
        t = r.turnaround_ms()
        stt = (r.text_transcribed - r.voice_input_sent) * 1000.0 if r.text_transcribed and r.voice_input_sent else None
        mock = (
            (r.mock_response_generated - r.text_transcribed) * 1000.0
            if r.mock_response_generated and r.text_transcribed
            else None
        )
        tts = (
            (r.first_audio_byte_received - r.mock_response_generated) * 1000.0
            if r.first_audio_byte_received and r.mock_response_generated
            else None
        )
        print(
            f"Run {i:02d} | STT: {stt:.1f}ms | Mock: {mock:.1f}ms | TTS First Byte: {tts:.1f}ms | E2E: {t:.1f}ms"
            if t and stt and mock and tts
            else f"Run {i:02d} | incomplete"
        )

    if not e2e_vals:
        print("No complete runs captured.")
        return

    p50 = percentile(e2e_vals, 50)
    p90 = percentile(e2e_vals, 90)
    p95 = percentile(e2e_vals, 95)
    ok = p95 <= 650.0

    print("\nSummary")
    print(f"p50: {p50:.1f}ms")
    print(f"p90: {p90:.1f}ms")
    print(f"p95: {p95:.1f}ms")
    print(f"median: {median(e2e_vals):.1f}ms")
    print(f"SLO (<=650ms): {'PASS' if ok else 'FAIL'}")


async def main() -> None:
    load_dotenv(".env.local")
    parser = argparse.ArgumentParser(description="Benchmark LiveKit turn latency")
    parser.add_argument("--pcm", required=True, help="Path to 5-second raw PCM16 16kHz mono file")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    livekit_url = os.getenv("LIVEKIT_URL", "")
    if not livekit_url:
        raise RuntimeError("LIVEKIT_URL is required")

    pcm_path = Path(args.pcm)
    if not pcm_path.exists():
        raise FileNotFoundError(str(pcm_path))

    results: list[RunMetrics] = []
    for i in range(1, args.runs + 1):
        result = await run_once(livekit_url, pcm_path, i)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
