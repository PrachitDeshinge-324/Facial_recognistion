"""Video processing utilities for face recognition system."""

from pathlib import Path
from typing import Tuple, Union

import cv2 as cv


def open_video(video_path: Union[str, int, Path]) -> cv.VideoCapture:
    """Open the video file or camera stream.
    
    Args:
        video_path: Path to video file or camera index
        
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    if isinstance(video_path, int):
        cap = cv.VideoCapture(video_path)
    else:
        cap = cv.VideoCapture(str(video_path))
        
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
    
    return cap


def create_video_writer(
    cap: cv.VideoCapture,
    output_path: Path | str,
    codec: str = "mp4v",
) -> cv.VideoWriter:
    """Create a video writer with appropriate parameters.
    
    Args:
        cap: Source video capture to get dimensions and FPS
        output_path: Path to save the output video
        codec: FourCC codec code
        
    Returns:
        OpenCV VideoWriter object
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    if width == 0 or height == 0:
        raise ValueError("Video capture has invalid frame dimensions")

    fourcc = cv.VideoWriter_fourcc(*codec)
    return cv.VideoWriter(str(output_path), fourcc, float(fps), (width, height))


def compute_fps_metrics(
    frame_count: int,
    total_fps: float,
    start_time: float,
    end_time: float,
) -> Tuple[float, float, float]:
    """Calculate FPS metrics.
    
    Args:
        frame_count: Number of frames processed
        total_fps: Running sum of FPS
        start_time: Start time of frame processing
        end_time: End time of frame processing
        
    Returns:
        Tuple of (current_fps, average_fps, updated_total_fps)
    """
    frame_duration = max(end_time - start_time, 1e-6)
    current_fps = 1.0 / frame_duration
    updated_total = total_fps + current_fps
    average_fps = updated_total / max(frame_count, 1)
    return current_fps, average_fps, updated_total