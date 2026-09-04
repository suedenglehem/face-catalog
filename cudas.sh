#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2 .v312/bin/python3 - <<'PY'
import torch

print("CUDA_VISIBLE_DEVICES =", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA device count    =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
