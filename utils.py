"""InsightFace utilities for face recognition system."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2 as cv
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config.models import AnalysisConfig, PRIMARY_MODEL, FALLBACK_MODEL, get_model_specific_providers


def create_face_analysis(config: AnalysisConfig | None = None) -> FaceAnalysis:
    """Create and prepare an InsightFace analysis instance with optimal device selection."""
    cfg = config or AnalysisConfig()
    
    def _init(model_name: str) -> FaceAnalysis:
        # Get model-specific providers for better compatibility
        providers = get_model_specific_providers(model_name)
        print(f"🔧 Using providers for {model_name}: {providers}")
        
        return FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=("detection", "recognition"),
        )

    models_to_try = [cfg.model_name]
    if cfg.model_name != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error: Exception | None = None
    successful_model = None
    used_providers = []
    
    for model_name in models_to_try:
        try:
            print(f"🤖 Initializing model: {model_name}")
            app = _init(model_name)
            successful_model = model_name
            used_providers = get_model_specific_providers(model_name)
            print(f"✅ Successfully loaded model: {model_name}")
            break
        except Exception as err:
            print(f"❌ Failed to load model {model_name}")
            if str(err).strip():  # Only print error if it's not empty
                print(f"   Error details: {err}")
            else:
                print(f"   Error: Model '{model_name}' not available or incompatible with current providers")
            last_error = err
    else:
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No compatible models could be loaded")

    # Automatically determine the best context ID based on the providers used
    ctx_id = _get_optimal_context_id(used_providers)
    print(f"🎯 Using context ID: {ctx_id}")
    
    app.prepare(ctx_id=ctx_id, det_size=cfg.det_size)
    return app


def _get_optimal_context_id(providers: list[str]) -> int:
    """Determine the optimal context ID based on available providers."""
    
    # Check for dedicated GPU providers that benefit from specific context IDs
    dedicated_gpu_providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider", 
        "ROCMExecutionProvider",
        "DirectMLExecutionProvider"
    ]
    
    # Check for Apple Silicon specific providers
    apple_providers = ["CoreMLExecutionProvider"]
    
    # If we have dedicated GPU providers, use GPU context
    if any(provider in providers for provider in dedicated_gpu_providers):
        print("🎮 Using GPU context (ID: 0) for dedicated GPU acceleration")
        return 0
    
    # For Apple Silicon with CoreML, let CoreML optimize automatically
    elif any(provider in providers for provider in apple_providers):
        print("🍎 Using CPU context (ID: -1) - CoreML will optimize across CPU/GPU/Neural Engine")
        return -1
    
    # For CPU-only execution
    else:
        print("💻 Using CPU context (ID: -1) for CPU-only execution")
        return -1


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
    """Create a cv.VideoWriter with sane fallbacks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    if width == 0 or height == 0:
        raise ValueError("Video capture has invalid frame dimensions")

    fourcc = cv.VideoWriter_fourcc(*codec)
    return cv.VideoWriter(str(output_path), fourcc, float(fps), (width, height))