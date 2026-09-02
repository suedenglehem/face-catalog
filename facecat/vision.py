from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis

from . import config


class FaceEngine:
    def __init__(self, ctx_id: int | None = None) -> None:
        self.ctx_id = config.GPU_DEVICE_ID if ctx_id is None else int(ctx_id)
        self.app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            # ORT requires one options dict per provider (same length as `providers`)
            provider_options=[{"device_id": self.ctx_id}, {}],
        )
        self.app.prepare(
            ctx_id=self.ctx_id,
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
                    # No separate quality model is loaded; det_score doubles as
                    # the quality proxy (group_cli picks group representatives
                    # by max quality_score).
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
