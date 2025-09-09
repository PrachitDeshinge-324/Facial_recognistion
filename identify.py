import os
import cv2
import time
import pandas as pd
import numpy as np
from deepface import DeepFace
from deepface.modules.verification import find_cosine_distance

# Load your face database
def load_database(file_path="face_database.pkl"):
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        print(f"Database file {file_path} not found.")
        return None

# Identify a face from an image frame
def identify_face(frame, database_df, model_name="Facenet512", detector_backend="retinaface"):
    try:
        # Convert frame to RGB (DeepFace uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        embedding_objs = DeepFace.represent(
            img_path=rgb_frame,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True  # Align faces for better accuracy
        )
        
        results = []
        for embedding_obj in embedding_objs:
            if "embedding" in embedding_obj:
                source_embedding = embedding_obj["embedding"]
                facial_area = embedding_obj["facial_area"]
                
                identities = []
                for idx, row in database_df.iterrows():
                    distance = find_cosine_distance(source_embedding, row["embedding"])
                    identities.append({
                        "identity": row["identity"],
                        "distance": distance,
                        "db_image_path": row["image_path"]
                    })
                
                if identities:
                    best_match = min(identities, key=lambda x: x["distance"])
                    
                    # Model-specific thresholds
                    thresholds = {
                        "VGG-Face": 0.68, 
                        "Facenet": 0.4, 
                        "Facenet512": 0.3,
                        "OpenFace": 0.1,
                        "DeepID": 0.015,
                        "ArcFace": 0.68,
                        "SFace": 0.593,
                        "GhostFaceNet": 0.65
                    }
                    
                    threshold = thresholds.get(model_name, 0.4)
                    verified = best_match["distance"] <= threshold
                    
                    # Calculate confidence score
                    if verified:
                        confidence = 100 * (1 - (best_match["distance"] / threshold))
                    else:
                        confidence = 100 * (threshold / best_match["distance"]) if best_match["distance"] > 0 else 0
                    
                    confidence = max(0, min(100, confidence))
                    
                    results.append({
                        "facial_area": facial_area,
                        "best_match": best_match["identity"],
                        "distance": best_match["distance"],
                        "threshold": threshold,
                        "verified": verified,
                        "confidence": confidence,
                        "db_image_path": best_match["db_image_path"]
                    })
        
        return results
        
    except Exception as e:
        print(f"Identification error: {e}")
        return []

# Main function for real-time face recognition
def real_time_face_recognition():
    # Configuration
    MODEL_NAME = "VGG-Face"        # You can try "SFace" for faster performance
    DETECTOR_BACKEND = "mtcnn"  # Best detector with facial landmarks
    
    # Load the face database
    db_df = load_database("face_database_{}.pkl".format(MODEL_NAME))
    if db_df is None:
        print("Please create the face database first.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # For FPS calculation
    prev_time = 0
    curr_time = 0
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Identify faces in the frame
        results = identify_face(frame, db_df, MODEL_NAME, DETECTOR_BACKEND)
        
        # Draw results on the frame
        for result in results:
            x, y, w, h = result["facial_area"]["x"], result["facial_area"]["y"], \
                         result["facial_area"]["w"], result["facial_area"]["h"]
            
            # Draw bounding box
            color = (0, 255, 0) if result["verified"] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            if result["verified"]:
                label = f"{result['best_match']} ({result['confidence']:.1f}%)"
            else:
                label = "Unknown"
            
            # Draw label background
            cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Real-Time Face Recognition', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_recognition()