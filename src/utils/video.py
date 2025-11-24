"""Video processing utilities for face recognition system."""

from pathlib import Path
from typing import Tuple, Union
import threading
import time

import cv2 as cv


def open_video(video_path: Union[str, int, Path], use_threading: bool = False) -> Union[cv.VideoCapture, 'ThreadedCamera']:
    """Open the video file or camera stream.
    
    Args:
        video_path: Path to video file or camera index
        use_threading: Whether to use threaded capture (recommended for cameras)
        
    Returns:
        OpenCV VideoCapture object or ThreadedCamera or None if failed
    """
    if use_threading:
        # Only import if needed to avoid circular imports if ThreadedCamera is in this file
        # (It is in this file, so we can use it directly)
        try:
            if isinstance(video_path, str) and video_path.isdigit():
                video_path = int(video_path)
            return ThreadedCamera(video_path)
        except Exception as e:
            print(f"Failed to start threaded camera: {e}, falling back to standard capture")

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
    processing_start_time: float,
    frame_start_time: float,
    frame_end_time: float
) -> Tuple[float, float]:
    """Calculate FPS metrics.
    
    Args:
        frame_count: Number of frames processed
        processing_start_time: Time when processing began (first frame)
        frame_start_time: Start time of current frame processing
        frame_end_time: End time of current frame processing
        
    Returns:
        Tuple of (current_fps, average_fps)
    """
    # Calculate current (instantaneous) FPS
    frame_duration = max(frame_end_time - frame_start_time, 1e-6)
    current_fps = 1.0 / frame_duration
    
    # Calculate true average FPS since beginning
    total_elapsed = frame_end_time - processing_start_time
    average_fps = frame_count / max(total_elapsed, 1e-6)
    
    return current_fps, average_fps


class ThreadedCamera:
    """Threaded camera capture for better performance."""
    
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 2)
        
        # FPS = 1/X
        # X = desired_fps
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.status = False
        self.frame = None
        self.stopped = False
        
        # Read first frame
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.FPS)

    def read(self):
        return self.status, self.frame
        
    def isOpened(self):
        return self.capture.isOpened()

    def get(self, propId):
        return self.capture.get(propId)

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()