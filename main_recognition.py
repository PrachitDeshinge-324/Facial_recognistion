"""Phase 2: Real-Time Face Recognition System.

This script performs real-time face recognition using the pre-built face
database. It now consumes shared helpers for configuration and InsightFace
initialisation to stay consistent with the rest of the codebase.
"""

from __future__ import annotations
from tqdm import tqdm

import pickle
import time
from pathlib import Path
from typing import Any, Union

import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from insightface.app import FaceAnalysis

from config.models import AnalysisConfig, get_recognition_config, get_model_config
from config.paths import OUTPUT_DATABASE, DEFAULT_VIDEO_PATH, OUTPUT_VIDEO_PATH
from utils import create_face_analysis, create_video_writer, compute_fps_metrics


class FaceRecognitionSystem:
    def __init__(
        self,
        database_file: Union[str, Path, None] = None,
        threshold: float | None = None,
    ) -> None:
        """Initialise the face recognition system."""

        model_config = get_model_config()
        recognition_config = get_recognition_config()

        analysis_config = AnalysisConfig(
            model_name=model_config["primary_model"],
            providers=tuple(model_config["providers"]),
            det_size=tuple(model_config["detection_size"]),
        )

        print("Initializing Face Recognition System...")
        self.app: FaceAnalysis = create_face_analysis(analysis_config)

        self.threshold = threshold or recognition_config["base_threshold"]
        print(f"Recognition threshold set to: {self.threshold}")

        self.database_path = Path(
            database_file or OUTPUT_DATABASE
        )
        self.face_database: dict[str, Any] | None = None
        self.load_face_database(self.database_path)

    def load_face_database(self, database_file):
        """Load the pre-built face database."""
        database_path = Path(database_file)
        if not database_path.exists():
            print(f"Warning: Face database file '{database_path}' not found!")
            print("Please run 'build_face_database.py' first to create the database.")
            print("Running in detection-only mode (no recognition).")
            return False
        
        try:
            with database_path.open('rb') as f:
                self.face_database = pickle.load(f)
            
            print(f"Face database loaded successfully!")
            print(f"Database contains {len(self.face_database['names'])} embeddings")
            print(f"Unique people: {len(set(self.face_database['names']))}")
            
            # Print database summary
            for name in set(self.face_database['names']):
                count = self.face_database['names'].count(name)
                print(f"  {name}: {count} embeddings")
            
            return True
            
        except Exception as e:
            print(f"Error loading face database: {str(e)}")
            print("Running in detection-only mode.")
            return False
    
    def recognize_face(self, face_embedding: np.ndarray):
        """
        Recognize a face by comparing its embedding with the database.
        
        Args:
            face_embedding: The 512-dimensional face embedding
            
        Returns:
            tuple: (recognized_name, confidence_score)
        """
        if self.face_database is None or len(self.face_database['embeddings']) == 0:
            return "Unknown", 0.0
        
        # Calculate cosine similarity with all database embeddings
        similarities = cosine_similarity([face_embedding], self.face_database['embeddings'])[0]
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Check if the best match exceeds the threshold
        if best_similarity >= self.threshold:
            recognized_name = self.face_database['names'][best_match_idx]
            return recognized_name, best_similarity
        else:
            return "Unknown", best_similarity
    
    def draw_face_info(self, frame, face, name, confidence):
        """
        Draw bounding box and recognition info on the frame.
        
        Args:
            frame: The video frame
            face: Face object from insightface
            name: Recognized name or "Unknown"
            confidence: Confidence score
        """
        # Get bounding box coordinates
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Choose colors based on recognition status
        if name == "Unknown":
            box_color = (0, 0, 255)  # Red for unknown
            text_color = (0, 0, 255)
        else:
            box_color = (0, 255, 0)  # Green for recognized
            text_color = (0, 255, 0)
        
        # Draw bounding box
        cv.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Prepare text
        if name == "Unknown":
            text = f"{name} ({confidence:.2f})"
        else:
            text = f"{name} ({confidence:.2f})"
        
        # Calculate text position
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
        
        # Draw text background
        cv.rectangle(frame, 
                    (text_x, text_y - text_size[1] - 5), 
                    (text_x + text_size[0] + 5, text_y + 5), 
                    box_color, -1)
        
        # Draw text
        cv.putText(frame, text, (text_x + 2, text_y - 2), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def open_video(video_path: Union[str, int, Path]) -> cv.VideoCapture | None:
    """Open the video file and return the video capture object."""

    if isinstance(video_path, int):
        cap = cv.VideoCapture(video_path)
    else:
        cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap

def display_fps(frame, current_fps, average_fps):
    """Overlay the FPS and Average FPS on the frame."""
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv.putText(frame, f"Avg FPS: {average_fps:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def process_video(cap: cv.VideoCapture, recognition_system: FaceRecognitionSystem, output_path: Path) -> None:
    """Process the video with face recognition."""
    frame_count = 0
    total_fps = 0
    
    print("\nStarting face recognition...")
    print("Press 'q' to quit")
    
    out = create_video_writer(cap, output_path)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if cap.get(cv.CAP_PROP_FRAME_COUNT) > 0 else None

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened() and frame_count < 500:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()
            frame_count += 1

            faces = recognition_system.app.get(frame)

            for face in faces:
                face_embedding = face.embedding
                name, confidence = recognition_system.recognize_face(face_embedding)
                recognition_system.draw_face_info(frame, face, name, confidence)

            current_time = time.perf_counter()
            current_fps, average_fps, total_fps = compute_fps_metrics(
                frame_count, total_fps, start_time, current_time
            )

            display_fps(frame, current_fps, average_fps)

            cv.putText(frame, f"Faces: {len(faces)}", (10, 90), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.putText(frame, f"Threshold: {recognition_system.threshold}", (10, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            out.write(frame)

            pbar.update(1)  # update tqdm progress bar

    out.release()


def main(video_path: Union[str, int, Path, None] = None) -> None:
    """Main function for the face recognition system."""
    print("=== Real-Time Face Recognition System ===")

    recognition_system = FaceRecognitionSystem()

    source = video_path if video_path is not None else DEFAULT_VIDEO_PATH

    # Open the video
    cap = open_video(source)
    if cap is None:
        print("Failed to open video source")
        return
    
    try:
        # Process the video
        process_video(cap, recognition_system, OUTPUT_VIDEO_PATH)
    finally:
        # Cleanup
        cap.release()
        # cv.destroyAllWindows()
        print("\nFace recognition system stopped.")

if __name__ == "__main__":
    main()
