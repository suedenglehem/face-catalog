# Face catalog app image: Python 3.12 + CUDA 12.x + cuDNN for onnxruntime-gpu.
# Ubuntu 24.04 ships python3.12 natively, so no separate python base is needed.
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps:
#  - python3-dev/build-essential/python3-pip: build wheels (rawpy, etc.)
#    and install packages (the CUDA base image ships no pip)
#  - python-is-python3: provides a `python` -> python3 symlink; the compose
#    index/group jobs call `python`, but Ubuntu only ships `python3`
#  - libglib2.0-0: required by Pillow at runtime
#  - exiftool: rich EXIF extraction (code falls back to PIL if absent)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev build-essential curl python3-pip python-is-python3 \
        libglib2.0-0t64 exiftool \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Ubuntu 24.04 marks the system Python as externally managed (PEP 668), so
# plain `pip install` refuses without --break-system-packages. The distro's
# pip 24.0 is used as-is: upgrading it fails because apt leaves no RECORD file.
RUN pip install --break-system-packages -r requirements.txt

COPY facecat ./facecat

# Non-root user. /data holds thumbnails (named volume); the insightface model
# cache lives at $HOME/.insightface and is also a named volume so models are
# downloaded once, not on every container start. UID 1001: the base image's
# `ubuntu` user already owns 1000.
RUN useradd -m -u 1001 app \
    && mkdir -p /data/thumbs /home/app/.insightface \
    && chown -R app:app /data /home/app/.insightface

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ || exit 1

CMD ["uvicorn", "facecat.webapp:app", "--host", "0.0.0.0", "--port", "8000"]