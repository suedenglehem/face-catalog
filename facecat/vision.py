from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis

from . import config


class FaceEngine:
    def __init__(self) -> None:
        self.app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(
            ctx_id=config.GPU_DEVICE_ID,
            det_size=(config.DET_SIZE, config.DET_SIZE),
        )

    def detect(self, rgb: np.ndarray) -> list[dict]:
        bgr = rgb[:, :, ::-1]
        faces = self.app.get(bgr)
        out = []

        for face in faces:
            bbox = [int(x) for x in face.bbox.tolist()]
            out.append(
                {
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "quality_score": float(face.det_score),
                    "embedding": np.asarray(face.normed_embedding, dtype=np.float32),
                }
            )

        return out

    def embedding_from_best_face(self, rgb: np.ndarray) -> np.ndarray:
        faces = self.detect(rgb)
        if not faces:
            raise ValueError("No face found in query image.")
        best = max(faces, key=lambda f: f["det_score"])
        return best["embedding"]
