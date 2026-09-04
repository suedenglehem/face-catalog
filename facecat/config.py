from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Environment variables come from the shell (source setenv.sh); no .env file is
# loaded here anymore.


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


def _cvd_map() -> dict[int, int] | None:
    """Map physical GPU index -> visible (CUDA/ORT) device id.

    Returns None when CUDA_VISIBLE_DEVICES is unset, in which case physical
    and visible ids coincide. Non-numeric entries (GPU UUIDs) count toward the
    visible total but cannot be mapped by index.
    """
    raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    mapping: dict[int, int] = {}
    for visible_id, entry in enumerate(e.strip() for e in raw.split(",") if e.strip()):
        try:
            mapping[int(entry)] = visible_id
        except ValueError:
            pass  # UUID form
    return mapping


def translate_gpus(physical_ids: list[int]) -> list[int]:
    """Translate physical GPU ids to ONNX Runtime device ids.

    ORT's `device_id` is a *visible* id (position within CUDA_VISIBLE_DEVICES)
    while nvidia-smi and user-facing flags speak physical indices; with CVD
    unset the two coincide. Hidden ids are skipped with a warning.
    """
    m = _cvd_map()
    if m is None:
        return list(dict.fromkeys(physical_ids))
    out: list[int] = []
    hidden: list[int] = []
    for p in physical_ids:
        if p in m:
            out.append(m[p])
        else:
            hidden.append(p)
    if hidden:
        print(
            f"[gpus] warning: physical GPU(s) {hidden} not visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}); skipping"
        )
    return sorted(set(out))


def _detect_gpus() -> list[int]:
    """Detect ONNX Runtime device ids to use (visible space).

    Priority: FACECAT_GPUS env var (physical indices, translated through
    CUDA_VISIBLE_DEVICES), then every visible device. nvidia-smi is only
    consulted when CVD is unset because it ignores CVD and always reports
    physical indices.
    """
    m = _cvd_map()

    def note(result: list[int]) -> list[int]:
        if m is not None:
            print(
                f"[gpus] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}: "
                f"using visible device ids {result}"
            )
        return result

    env = os.getenv("FACECAT_GPUS")
    if env:
        try:
            idxs = [int(x) for x in env.split(",") if x.strip()]
        except ValueError:
            idxs = []
        if idxs:
            translated = translate_gpus(idxs)
            if not translated:
                raise ValueError(
                    f"FACECAT_GPUS={env!r}: every requested GPU is hidden by CUDA_VISIBLE_DEVICES"
                )
            return note(translated)

    if m is not None:
        # CVD set: visible ids are exactly 0..n-1 by construction, so no
        # nvidia-smi round trip is needed (and its physical indices would lie).
        n = len([e for e in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if e.strip()])
        return note(list(range(n)))

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
# VRAM kept free per GPU in --threads-max mode, in GB (decimals allowed).
RESERVE_VRAM_GB = float(os.getenv("RESERVE_VRAM", "2"))
INDEX_CPU_WORKERS = int(os.getenv("INDEX_CPU_WORKERS", "8"))
INDEX_DECODE_QUEUE = int(os.getenv("INDEX_DECODE_QUEUE", "16"))
INDEX_GPU_QUEUE = int(os.getenv("INDEX_GPU_QUEUE", "8"))

# --- group rebuild (group_cli) ---
GROUP_BLOCK_SIZE = int(os.getenv("GROUP_BLOCK_SIZE", "512"))
KNN_CHUNK = int(os.getenv("KNN_CHUNK", "0"))  # 0 = adaptive per collection size
