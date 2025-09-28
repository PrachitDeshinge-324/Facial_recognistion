"""Standalone detection script using the InsightFace detector only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import cv2 as cv

from insightface.app import FaceAnalysis

from .config_utils import load_config
from .insightface_utils import AnalysisConfig, create_face_analysis, compute_fps_metrics

DEFAULT_VIDEO_PATH = Path("../Facial Recognision/video/03_09_2025_face_recognition.mp4")

def open_video(video_path: Union[str, int, Path]) -> cv.VideoCapture | None:
    """Open the video source and return a capture object."""

    if isinstance(video_path, int):
        cap = cv.VideoCapture(video_path)
    else:
        cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return None
    return cap


def display_fps(frame, current_fps, average_fps):
    """Overlay the FPS and Average FPS on the frame."""
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Avg FPS: {average_fps:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def process_video(cap: cv.VideoCapture, app: FaceAnalysis) -> None:
    """Process the video, frame by frame, and display FPS."""
    frame_count = 0
    total_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Start of processing for this frame ---
        start_time = time.perf_counter()
        
        frame_count += 1

        # Detect faces in the image
        faces = app.get(frame)

        # Draw bounding boxes on the faces
        rimg = app.draw_on(frame, faces)

        # --- End of processing for this frame ---
        current_time = time.perf_counter()

        current_fps, average_fps, total_fps = compute_fps_metrics(
            frame_count, total_fps, start_time, current_time
        )

        # Overlay FPS on the result image (rimg)
        display_fps(rimg, current_fps, average_fps)

        # Show the frame with detections and FPS overlay
        cv.imshow('Video with FPS', rimg)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Use waitKey(1) for smoother video playback
            break


def main(video_path: Union[str, int, Path, None] = None) -> None:
    config = load_config()

    model_config = config.get("model_config", {})
    defaults = AnalysisConfig()
    analysis_config = AnalysisConfig(
        model_name=model_config.get("fallback_model", "buffalo_l"),
        providers=tuple(model_config.get("providers", defaults.providers)),
        det_size=tuple(model_config.get("detection_size", defaults.det_size)),
        ctx_id=model_config.get("ctx_id", 0),
    )

    app = create_face_analysis(analysis_config)

    source = video_path if video_path is not None else DEFAULT_VIDEO_PATH

    # Open the video
    cap = open_video(source)
    if cap is None:
        return

    # Process the video
    process_video(cap, app)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
