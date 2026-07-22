"""Small, dependency-light helpers for structured voice-turn latency logs."""
import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager

logger = logging.getLogger("voice-agent.latency")


def resource_snapshot() -> dict:
    result: dict = {}
    try:
        import psutil
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        result.update(
            cpu_percent=psutil.cpu_percent(interval=None),
            process_rss_mb=round(proc.memory_info().rss / 1048576, 1),
            ram_used_percent=vm.percent,
        )
    except Exception:
        pass
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1, check=False,
        )
        if query.returncode == 0 and query.stdout.strip():
            gpu, used, total = [int(x.strip()) for x in query.stdout.splitlines()[0].split(",")]
            result.update(gpu_percent=gpu, vram_used_mb=used, vram_total_mb=total)
    except Exception:
        pass
    return result


def log_stage(stage: str, duration_ms: float | None = None, **fields) -> None:
    event = {"event": "latency", "stage": stage, **fields}
    if duration_ms is not None:
        event["duration_ms"] = round(duration_ms, 2)
    event.update(resource_snapshot())
    logger.info(json.dumps(event, ensure_ascii=False, default=str))


@contextmanager
def timed(stage: str, **fields):
    started = time.perf_counter()
    try:
        yield
    finally:
        log_stage(stage, (time.perf_counter() - started) * 1000, **fields)
