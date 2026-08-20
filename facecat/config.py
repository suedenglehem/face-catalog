from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = os.environ["DATABASE_URL"]

THUMBS_DIR = Path(os.getenv("THUMBS_DIR", "./data/thumbs")).resolve()
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "buffalo_l")
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))
DET_SIZE = int(os.getenv("DET_SIZE", "640"))
MAX_DETECT_SIDE = int(os.getenv("MAX_DETECT_SIDE", "1600"))

ALLOW_CPU_RESIZE = _bool("ALLOW_CPU_RESIZE", True)

GROUP_K = int(os.getenv("GROUP_K", "20"))
GROUP_THRESHOLD = float(os.getenv("GROUP_THRESHOLD", "0.55"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "50"))
