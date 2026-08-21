from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

from . import config

RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf", ".pef"
}

IMAGE_EXTS = RAW_EXTS | {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def load_rgb(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in RAW_EXTS:
        with rawpy.imread(str(path)) as raw:
            # Note: rawpy >= 0.20 removed the `auto_bright` kwarg (default off).
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
            )
        return rgb

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def extract_exif(path: Path) -> dict:
    if shutil.which("exiftool"):
        cp = subprocess.run(
            ["exiftool", "-json", "-n", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(cp.stdout)[0]
        keep = {
            k: data[k]
            for k in (
                "DateTimeOriginal",
                "CreateDate",
                "Model",
                "Make",
                "LensModel",
                "FocalLength",
                "FNumber",
                "ISO",
                "ExposureTime",
                "ImageWidth",
                "ImageHeight",
                "GPSLatitude",
                "GPSLongitude",
            )
            if k in data
        }
        return keep

    try:
        with Image.open(path) as im:
            raw = im.getexif()
            return {str(k): raw.get(k) for k in raw.keys()}
    except Exception:
        return {}


def resize_for_detector(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return rgb, 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gpu = cv2.cuda_GpuMat()
        gpu.upload(bgr)
        gpu_out = cv2.cuda.resize(gpu, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out_bgr = gpu_out.download()
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return out_rgb, 1.0 / scale

    if not config.ALLOW_CPU_RESIZE:
        raise RuntimeError(
            "CUDA resize requested, but no CUDA-enabled OpenCV build is available. "
            "Install OpenCV with CUDA or set ALLOW_CPU_RESIZE=1."
        )

    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out, 1.0 / scale


def clamp_box(box: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def crop_face(rgb: np.ndarray, box: list[int], pad_frac: float = 0.15) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_frac)
    py = int(bh * pad_frac)

    box2 = clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], w, h)
    a, b, c, d = box2
    return rgb[b:d, a:c]


def save_jpeg(rgb: np.ndarray, out_path: Path, size: tuple[int, int] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb)
    if size is not None:
        im.thumbnail(size)
    im.save(out_path, format="JPEG", quality=90)
