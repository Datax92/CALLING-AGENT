# Summary of changes made to fix the Railway build:

## 1. Dockerfile (consolidated)
- Changed base image from python:3.9-slim to python:3.12-slim in both builder and final stages.
- Removed the installation of mongodb-org (since MongoDB is provided as a managed service).
- Updated the COPY command to copy from python3.12 site-packages.
- Changed the CMD to run uvicorn directly (JSON array form) instead of trying to start MongoDB and then the app.

## 2. Dockerfile.voice
- Updated base image to python:3.12-slim.
- Removed build-essential from the runtime stage (only needed in builder).
- Updated COPY to use python3.12 site-packages.
- Removed the step that downloads the Piper TTS voice (now mounted as a volume).
- Kept the CMD to run agent.py.

## 3. Dockerfile.dashboard
- Updated base image to python:3.12-slim.
- Updated COPY to use python3.12 site-packages.
- Changed CMD to run uvicorn with the app (JSON array form).

## 4. requirements-voice.txt
- Removed vllm==0.5.3 and sglang==0.1.15 (the voice agent only talks to the vLLM server via HTTP).
- Removed livekit-plugins-turn-detector==1.6.4 (not needed for outbound calls).
- Changed requests from 2.34.2 to 2.32.3 (since 2.34.2 was yanked from PyPI).
- Kept all other dependencies (livekit-agents, livekit-plugins-deepgram, piper-tts, etc.).

## 5. requirements-dashboard.txt
- Changed pymongo from 4.8.0 to 4.9.0 to satisfy the dependency of motor==3.6.0 (which requires pymongo>=4.9,<4.10).

## 6. .dockerignore
- Removed the problematic line that excluded all *.txt files (which was preventing the requirement files from being copied).
- Kept all other exclusions (logs, caches, model directories, etc.) to keep the build context small.

These changes should resolve all the build errors encountered on Railway:
- Python version mismatch (now 3.12 satisfies livekit-agents>=3.10,<3.15)
- Missing mongodb-org package (removed the install step)
- Dependency conflict between motor and pymongo (fixed by aligning versions)
- Missing requirement files in build context (fixed .dockerignore)
- Uninstallable vllm-flash-attin dependency (removed vllm and sglang from voice agent)

After applying these changes, the Docker image should build successfully and the services (voice agent, dashboard, and the separate vLLM server) should start correctly.