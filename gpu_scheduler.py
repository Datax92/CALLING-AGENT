import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger("voice-agent.gpu-scheduler")

class GPUScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.current_pod: Optional[str] = None
        self.next_pod: Optional[str] = None
        self.runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
        self.peak_pod_id = os.getenv("PEAK_POD_ID", "")
        self.offpeak_pod_id = os.getenv("OFFPEAK_POD_ID", "")
        self.peak_window_start = os.getenv("PEAK_WINDOW_START", "09:00")
        self.peak_window_end = os.getenv("PEAK_WINDOW_END", "21:00")
        self.handoff_buffer_minutes = int(os.getenv("HANDOFF_BUFFER_MINUTES", "30"))

    async def start(self):
        """Start the scheduler with initial pod selection and scheduled jobs."""
        # Determine initial pod based on current time
        self._select_initial_pod()

        # Schedule pod switching jobs
        self.scheduler.add_job(
            self._check_and_switch_pods,
            'interval',
            minutes=15,
            next_run_time=datetime.now()
        )

        # Schedule pod health checks
        self.scheduler.add_job(
            self._check_pod_health,
            'interval',
            minutes=5,
            next_run_time=datetime.now() + timedelta(minutes=2)
        )

        self.scheduler.start()
        logger.info(f"GPU scheduler started with initial pod: {self.current_pod}")

    def _select_initial_pod(self):
        """Select the initial pod based on current time."""
        now = datetime.now()
        peak_start = datetime.strptime(self.peak_window_start, "%H:%M").time()
        peak_end = datetime.strptime(self.peak_window_end, "%H:%M").time()
        current_time = now.time()

        if peak_start <= current_time <= peak_end:
            self.current_pod = self.peak_pod_id
            self.next_pod = self.offpeak_pod_id
        else:
            self.current_pod = self.offpeak_pod_id
            self.next_pod = self.peak_pod_id

        logger.info(f"Initial pod selected: {self.current_pod}")

    async def _check_and_switch_pods(self):
        """Check if pods need to be switched based on time and call volume."""
        now = datetime.now()
        peak_start = datetime.strptime(self.peak_window_start, "%H:%M").time()
        peak_end = datetime.strptime(self.peak_window_end, "%H:%M").time()
        current_time = now.time()

        # Check if we're in peak or off-peak window
        if peak_start <= current_time <= peak_end:
            target_pod = self.peak_pod_id
        else:
            target_pod = self.offpeak_pod_id

        # Check if we need to switch pods
        if self.current_pod != target_pod:
            logger.info(f"Switching from pod {self.current_pod} to {target_pod}")
            await self._switch_pods(target_pod)

    async def _switch_pods(self, target_pod: str):
        """Switch from current pod to target pod."""
        try:
            # Stop the current pod
            if self.current_pod:
                await self._stop_pod(self.current_pod)

            # Start the target pod
            await self._start_pod(target_pod)

            # Update current and next pod
            self.current_pod = target_pod
            self.next_pod = self.peak_pod_id if target_pod == self.offpeak_pod_id else self.offpeak_pod_id

            logger.info(f"Successfully switched to pod {target_pod}")
        except Exception as e:
            logger.error(f"Failed to switch pods: {e}")

    async def _start_pod(self, pod_id: str):
        """Start a RunPod."""
        if not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY is not set")

        url = f"https://api.runpod.io/v2/pod/{pod_id}/start"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers)
        response.raise_for_status()
        logger.info(f"Started pod {pod_id}")

    async def _stop_pod(self, pod_id: str):
        """Stop a RunPod."""
        if not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY is not set")

        url = f"https://api.runpod.io/v2/pod/{pod_id}/stop"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers)
        response.raise_for_status()
        logger.info(f"Stopped pod {pod_id}")

    async def _check_pod_health(self):
        """Check the health of the current pod and restart if needed."""
        if not self.current_pod:
            return

        try:
            url = f"https://api.runpod.io/v2/pod/{self.current_pod}/status"
            headers = {
                "Authorization": f"Bearer {self.runpod_api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = response.json().get("data", {}).get("status")

            if status != "RUNNING":
                logger.warning(f"Pod {self.current_pod} is not running (status: {status}). Attempting to restart...")
                await self._start_pod(self.current_pod)
        except Exception as e:
            logger.error(f"Failed to check pod health: {e}")

# Global scheduler instance
scheduler = GPUScheduler()

async def start_scheduler():
    """Start the GPU scheduler."""
    await scheduler.start()

async def stop_scheduler():
    """Stop the GPU scheduler."""
    scheduler.scheduler.shutdown()
    logger.info("GPU scheduler stopped")