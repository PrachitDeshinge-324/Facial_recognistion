"""Shared helpers for InsightFace-based pipelines.

These utilities consolidate repeated setup logic so that the detection,
recognition, and database-building scripts stay lean and consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2 as cv
from insightface.app import FaceAnalysis

DEFAULT_MODEL_NAME = "antelopev2"
DEFAULT_DET_SIZE: Tuple[int, int] = (640, 640)
DEFAULT_PROVIDERS: Tuple[str, ...] = (
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration needed to prepare an InsightFace analysis instance."""

    model_name: str = DEFAULT_MODEL_NAME
    providers: Sequence[str] = DEFAULT_PROVIDERS
    det_size: Tuple[int, int] = DEFAULT_DET_SIZE
    ctx_id: int = 0


def create_face_analysis(config: AnalysisConfig | None = None) -> FaceAnalysis:
    """Initialise and prepare a :class:`FaceAnalysis` instance."""

    cfg = config or AnalysisConfig()
    app = FaceAnalysis(name=cfg.model_name, providers=list(cfg.providers))
    app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size)
    return app


def compute_fps_metrics(
    frame_count: int,
    total_fps: float,
    start_time: float,
    end_time: float,
) -> tuple[float, float, float]:
    """Return instantaneous and rolling FPS metrics for a frame."""

    frame_duration = max(end_time - start_time, 1e-6)
    current_fps = 1.0 / frame_duration
    updated_total = total_fps + current_fps
    average_fps = updated_total / max(frame_count, 1)
    return current_fps, average_fps, updated_total


def create_video_writer(
    cap: cv.VideoCapture,
    output_path: Path | str,
    codec: str = "mp4v",
) -> cv.VideoWriter:
    """Create a :class:`cv.VideoWriter` with sane fallbacks."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    if width == 0 or height == 0:
        raise ValueError("Video capture has invalid frame dimensions")

    fourcc = cv.VideoWriter_fourcc(*codec)
    return cv.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
