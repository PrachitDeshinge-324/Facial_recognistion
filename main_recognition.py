"""
Phase 2: Real-Time Face Recognition System
This script performs real-time face recognition using the pre-built face database.
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionSystem:
    def __init__(self, database_file="database/face_database_antelopev2.pkl", threshold=0.5):
        """
        Initialize the face recognition system.
        
        Args:
            database_file: Path to the pickled face database
            threshold: Cosine similarity threshold for recognition (0.5-0.7 recommended)
        """
        print("Initializing Face Recognition System...")
        
        # Initialize the face analysis model
        self.app = FaceAnalysis(name='antelopev2', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Set recognition threshold
        self.threshold = threshold
        print(f"Recognition threshold set to: {threshold}")
        
        # Load the face database
        self.face_database = None
        self.load_face_database(database_file)
        
    def load_face_database(self, database_file):
        """Load the pre-built face database."""
        if not os.path.exists(database_file):
            print(f"Warning: Face database file '{database_file}' not found!")
            print("Please run 'build_face_database.py' first to create the database.")
            print("Running in detection-only mode (no recognition).")
            return False
        
        try:
            with open(database_file, 'rb') as f:
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
    
    def recognize_face(self, face_embedding):
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

def open_video(video_path):
    """Open the video file and return the video capture object."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap

def display_fps(frame, current_fps, average_fps):
    """Overlay the FPS and Average FPS on the frame."""
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv.putText(frame, f"Avg FPS: {average_fps:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def process_video(cap, recognition_system):
    """Process the video with face recognition."""
    frame_count = 0
    total_fps = 0
    
    print("\nStarting face recognition...")
    print("Press 'q' to quit")
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Define video writer
    output_path = os.path.join('output', 'output_video.mp4')
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    fps = 30  # Set to your video's FPS
    frame_size = (width, height)  # Set to your frame size (width, height)

    out = cv.VideoWriter(output_path, fourcc, fps, frame_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing for this frame
        start_time = time.time()
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
        current_time = time.time()
        processing_time = current_time - start_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        
        total_fps += current_fps
        average_fps = total_fps / frame_count
        
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

def main():
    """Main function for the face recognition system."""
    print("=== Real-Time Face Recognition System ===")
    
    # Initialize the recognition system
    recognition_system = FaceRecognitionSystem(threshold=0.2)
    
    # Video path - you can change this or use webcam (0)
    video_path = '../Facial Recognision/video/03_09_2025_face_recognition.mp4'
    # video_path = '../Person Identification/v_1/input/3c.mp4'
    # video_path = 0
    # For webcam, use: video_path = 0
    
    # Open the video
    cap = open_video(video_path)
    if cap is None:
        print("Failed to open video source")
        return
    
    try:
        # Process the video
        process_video(cap, recognition_system)
    finally:
        # Cleanup
        cap.release()
        cv.destroyAllWindows()
        print("\nFace recognition system stopped.")

if __name__ == "__main__":
    main()
