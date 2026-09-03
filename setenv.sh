# Activate the project venv and export facecat environment variables.
# This file is the single source of truth for host configuration (the old `env`
# and `.env` files are gone). Source it, don't run it:   . ./setenv.sh [GPUS]
#
# Optional first argument sets CUDA_VISIBLE_DEVICES for this session, overriding
# any value already in the environment. Omit it to keep whatever is currently set
# (or leave every GPU visible when unset). When CUDA_VISIBLE_DEVICES ends up set --
# via the argument or a pre-existing environment value -- the cards we will run on
# are printed at the end of this script.

# Directory of this file. BASH_SOURCE is set when sourced under bash; it is empty
# under dash (make), where we fall back to the current directory -- fine, because
# make runs from the repo root. No array subscript so dash accepts it.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE:-.}")" && pwd)"
. "$REPO_DIR/.v312/bin/activate"

# --- facecat configuration ---------------------------------------------------
set -a
DATABASE_URL="postgresql://facecat:facecat@127.0.0.1:5432/facecat"
THUMBS_DIR="$REPO_DIR/thumbs"
MODEL_NAME="buffalo_l"
GPU_DEVICE_ID=0
DET_SIZE=640
MAX_DETECT_SIDE=1600

ALLOW_CPU_RESIZE=1

GROUP_K=20
GROUP_THRESHOLD=0.55
SEARCH_LIMIT=50

# --- multi-GPU indexing / grouping (defaults; CLI flags override) ---
THREADS_PER_GPU=4
INDEX_CPU_WORKERS=8
INDEX_DECODE_QUEUE=16
INDEX_GPU_QUEUE=8
GROUP_BLOCK_SIZE=512
KNN_CHUNK=0
set +a

# cuDNN 9 ships in the venv as a wheel (nvidia-cudnn-cu12); CUDA 12 runtime is
# system-wide. Both are needed by onnxruntime-gpu at import time, otherwise it
# silently falls back to CPU. Anchored to REPO_DIR so this works from any cwd.
export LD_LIBRARY_PATH="$REPO_DIR/.v312/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12/lib64:${LD_LIBRARY_PATH:-}"

# --- CUDA_VISIBLE_DEVICES ----------------------------------------------------
# Optional first argument wins; otherwise keep whatever the environment already
# has (a previously exported value survives re-sourcing -- `unset` it or open a
# new shell to clear). This box: 0 RTX 3090, 1 RTX 3080 Ti, 2 RTX 3090.
if [ -n "${1:-}" ]; then
  CUDA_VISIBLE_DEVICES="$1"
fi
# Re-export when set (from argument or pre-existing env) so child processes such
# as python see it even if the value only arrived via a command prefix in make.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
fi

# Report which physical cards we will run on when CVD is set (via argument or a
# pre-existing environment value). nvidia-smi maps each index to its card name;
# if it is unavailable the raw indices are printed instead.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _gpu_table="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)"
  cards=""
  for idx in $(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
    [ -z "$idx" ] && continue
    name="$(printf '%s\n' "$_gpu_table" | awk -F', ' -v i="$idx" '$1==i {print $2; exit}')"
    if [ -n "$name" ]; then
      cards="${cards}${cards:+, }${idx}: ${name}"
    else
      cards="${cards}${cards:+, }${idx}"
    fi
  done
  echo "[setenv] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> will run on GPU(s): ${cards}"
fi
