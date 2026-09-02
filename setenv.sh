# Activate the project venv and export facecat environment variables.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.v312/bin/activate"
set -a; source "$REPO_DIR/env"; set +a

# Re-anchor LD_LIBRARY_PATH to the repo dir (env uses $PWD, which is wrong if
# this file is sourced from another directory). cuDNN 9 comes from the
# nvidia-cudnn-cu12 wheel in the venv; CUDA 12 runtime is system-wide.
export LD_LIBRARY_PATH="$REPO_DIR/.v312/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12/lib64:${LD_LIBRARY_PATH:-}"
