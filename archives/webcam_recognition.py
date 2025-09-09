"""
Webcam Face Recognition
Real-time face recognition using webcam feed.
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class WebcamFaceRecognition:
    def __init__(self, database_file="face_database.pkl", threshold=0.6):
        """Initialize webcam face recognition system."""
        print("Initializing Webcam Face Recognition...")
        
        # Initialize the face analysis model
        self.app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Set recognition threshold
        self.threshold = threshold
        
        # Load the face database
        self.face_database = None
        self.load_face_database(database_file)
        
    def load_face_database(self, database_file):
        """Load the pre-built face database."""
        if not os.path.exists(database_file):
            print(f"Warning: Face database file '{database_file}' not found!")
            print("Please run 'build_face_database.py' first.")
            return False
        
        try:
            with open(database_file, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"Face database loaded: {len(set(self.face_database['names']))} people")
            return True
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return False
    
    def recognize_face(self, face_embedding):
        """Recognize a face using cosine similarity."""
        if self.face_database is None:
            return "Unknown", 0.0
        
        similarities = cosine_similarity([face_embedding], self.face_database['embeddings'])[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= self.threshold:
            return self.face_database['names'][best_match_idx], best_similarity
        else:
            return "Unknown", best_similarity
    
    def draw_results(self, frame, face, name, confidence):
        """Draw recognition results on frame."""
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Colors
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw box
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text
        text = f"{name} ({confidence:.2f})"
        cv.putText(frame, text, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def run(self):
        """Run the webcam recognition system."""
        cap = cv.VideoCapture(0)  # Use default webcam
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        
        print("Webcam face recognition started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv.flip(frame, 1)
            
            # Detect faces
            faces = self.app.get(frame)
            
            # Process each face
            for face in faces:
                name, confidence = self.recognize_face(face.embedding)
                self.draw_results(frame, face, name, confidence)
            
            # Display info
            cv.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.putText(frame, f"Threshold: {self.threshold}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv.imshow('Webcam Face Recognition', frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()

def main():
    """Main function."""
    print("=== Webcam Face Recognition ===")
    
    # You can adjust the threshold here (0.5-0.7 recommended)
    recognition_system = WebcamFaceRecognition(threshold=0.6)
    recognition_system.run()

if __name__ == "__main__":
    main()
