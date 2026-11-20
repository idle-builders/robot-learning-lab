# Local CPU-only replay on Apple Silicon through Docker's x86 emulation.
# No training libraries or provider credentials.
FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
RUN python -m pip install --disable-pip-version-check --no-input --no-cache-dir \
    spectacularAI==1.54.0 numpy==2.2.6 imageio-ffmpeg==0.6.0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
