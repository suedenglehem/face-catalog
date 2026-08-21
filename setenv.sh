set -a; source ./env; set +a
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda-12.5/bin:${PATH}