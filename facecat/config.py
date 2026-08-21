from __future__ import annotations

import os
import shutil
import subprocess
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


def _detect_gpus() -> list[int]:
    """Detect available GPU indices.

    Priority: FACECAT_GPUS env var, then nvidia-smi, then [GPU_DEVICE_ID].
    Indices are physical; keep CUDA_VISIBLE_DEVICES unset or "0,1" so they
    match the device ids passed to ONNX Runtime.
    """
    env = os.getenv("FACECAT_GPUS")
    if env:
        try:
            idxs = [int(x) for x in env.split(",") if x.strip()]
            if idxs:
                return idxs
        except ValueError:
            pass
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            idxs = [int(line.strip()) for line in out.splitlines() if line.strip()]
            if idxs:
                return idxs
        except Exception:
            pass
    return [GPU_DEVICE_ID]


GPUS = _detect_gpus()

# --- indexing pipeline (index_cli) ---
THREADS_PER_GPU = int(os.getenv("THREADS_PER_GPU", "4"))
INDEX_CPU_WORKERS = int(os.getenv("INDEX_CPU_WORKERS", "8"))
INDEX_DECODE_QUEUE = int(os.getenv("INDEX_DECODE_QUEUE", "16"))
INDEX_GPU_QUEUE = int(os.getenv("INDEX_GPU_QUEUE", "8"))

# --- group rebuild (group_cli) ---
GROUP_BLOCK_SIZE = int(os.getenv("GROUP_BLOCK_SIZE", "512"))
KNN_CHUNK = int(os.getenv("KNN_CHUNK", "0"))  # 0 = adaptive per collection size
