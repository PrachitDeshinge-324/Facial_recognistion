"""Phase 2: Real-Time Face Recognition System.

This script performs real-time face recognition using the pre-built face
database. It now consumes shared helpers for configuration and InsightFace
initialisation to stay consistent with the rest of the codebase.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Union

import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from insightface.app import FaceAnalysis

from src.config_utils import load_config
from src.insightface_utils import (
    AnalysisConfig,
    create_face_analysis,
    create_video_writer,
    compute_fps_metrics,
)

DEFAULT_VIDEO_PATH = Path("../Facial Recognision/video/03_09_2025_face_recognition.mp4")


class FaceRecognitionSystem:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        database_file: Union[str, Path, None] = None,
        threshold: float | None = None,
    ) -> None:
        """Initialise the face recognition system."""

        config = config or {}
        model_config = config.get("model_config", {})
        recognition_config = config.get("recognition_config", {})
        path_config = config.get("paths", {})

        defaults = AnalysisConfig()
        analysis_config = AnalysisConfig(
            model_name=model_config.get("primary_model", defaults.model_name),
            providers=tuple(model_config.get("providers", defaults.providers)),
            det_size=tuple(model_config.get("detection_size", defaults.det_size)),
            ctx_id=model_config.get("ctx_id", defaults.ctx_id),
        )

        print("Initializing Face Recognition System...")
        self.app: FaceAnalysis = create_face_analysis(analysis_config)

        self.threshold = threshold or recognition_config.get("base_threshold", 0.4)
        print(f"Recognition threshold set to: {self.threshold}")

        self.database_path = Path(
            database_file
            or path_config.get("output_database")
            or "database/face_database_antelopev2.pkl"
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
    
    # Ensure output directory exists
    out = create_video_writer(cap, output_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing for this frame
        start_time = time.perf_counter()
        frame_count += 1
        
        # Detect faces in the frame
        faces = recognition_system.app.get(frame)
        
        # Process each detected face
        for face in faces:
            # Get the face embedding
            face_embedding = face.embedding
            
            # Recognize the face
            name, confidence = recognition_system.recognize_face(face_embedding)
            
            # Draw the face info on the frame
            recognition_system.draw_face_info(frame, face, name, confidence)
        
        # Calculate FPS
        current_time = time.perf_counter()
        current_fps, average_fps, total_fps = compute_fps_metrics(
            frame_count, total_fps, start_time, current_time
        )
        
        # Display FPS
        display_fps(frame, current_fps, average_fps)
        
        # Display system info
        cv.putText(frame, f"Faces: {len(faces)}", (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv.putText(frame, f"Threshold: {recognition_system.threshold}", (10, 120), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)

        # Show the frame
        cv.imshow('Face Recognition System', frame)
        
        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video writer
    out.release()

def main(video_path: Union[str, int, Path, None] = None) -> None:
    """Main function for the face recognition system."""
    print("=== Real-Time Face Recognition System ===")

    config = load_config()
    recognition_system = FaceRecognitionSystem(config=config)

    source = video_path if video_path is not None else DEFAULT_VIDEO_PATH

    # Open the video
    cap = open_video(source)
    if cap is None:
        print("Failed to open video source")
        return
    
    try:
        output_config = config.get("paths", {})
        output_path = Path(output_config.get("recognition_output", "output/output_video.mp4"))
        # Process the video
        process_video(cap, recognition_system, output_path)
    finally:
        # Cleanup
        cap.release()
        cv.destroyAllWindows()
        print("\nFace recognition system stopped.")

if __name__ == "__main__":
    main()
