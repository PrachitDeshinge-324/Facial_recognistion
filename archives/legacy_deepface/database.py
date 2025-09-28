import os
import cv2
import pandas as pd
from deepface import DeepFace
from deepface.modules.verification import find_cosine_distance

def build_face_database(database_path, model_name="Facenet512", detector_backend="retinaface"):
    """
    Build a facial recognition database from folder structure
    
    Args:
        database_path (str): Path to the root folder containing subfolders of people
        model_name (str): Which DeepFace model to use for facial embeddings
        detector_backend (str): Which backend to use for face detection
    
    Returns:
        pandas.DataFrame: Database with facial embeddings and metadata
    """
    print(f"Building face database using {model_name} model...")
    
    # Initialize database
    db_df = pd.DataFrame(columns=["identity", "embedding", "image_path"])
    
    # Iterate through each person's folder
    for person_name in os.listdir(database_path):
        person_path = os.path.join(database_path, person_name)
        
        if os.path.isdir(person_path):
            print(f"Processing {person_name}...")
            
            # Process each image in the person's folder
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    
                    try:
                        # Get facial embedding
                        embedding_objs = DeepFace.represent(
                            img_path=img_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        
                        # Add each found face to database
                        for embedding_obj in embedding_objs:
                            new_row = {
                                "identity": person_name,
                                "embedding": embedding_obj["embedding"],
                                "image_path": img_path
                            }
                            db_df = pd.concat([db_df, pd.DataFrame([new_row])], ignore_index=True)
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Database built with {len(db_df)} facial embeddings.")
    return db_df

def identify_face(img_path, database_df, model_name="Facenet512", detector_backend="retinaface"):
    """
    Identify a face in an image by comparing against the database
    
    Args:
        img_path (str): Path to image with face to identify
        database_df (pandas.DataFrame): Database with facial embeddings
        model_name (str): Which DeepFace model to use
        detector_backend (str): Which backend to use for face detection
    
    Returns:
        dict: Identification results
    """
    print(f"Identifying face in {img_path}...")
    
    try:
        # Find the face in the image
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        
        if not embedding_objs:
            return {"error": "No faces detected in the image"}
        
        results = []
        for embedding_obj in embedding_objs:
            source_embedding = embedding_obj["embedding"]
            facial_area = embedding_obj["facial_area"]
            
            # Compare with all embeddings in database
            identities = []
            for idx, row in database_df.iterrows():
                distance = find_cosine_distance(source_embedding, row["embedding"])
                identities.append({
                    "identity": row["identity"],
                    "distance": distance,
                    "db_image_path": row["image_path"]
                })
            
            # Find the best match (smallest distance)
            if identities:
                best_match = min(identities, key=lambda x: x["distance"])
                
                # Use model-specific threshold (approximate values)
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
                
                # Calculate confidence score (inverse of distance relative to threshold)
                if verified:
                    confidence = 100 * (1 - (best_match["distance"] / threshold))
                else:
                    confidence = 100 * (threshold / best_match["distance"]) if best_match["distance"] > 0 else 0
                
                confidence = max(0, min(100, confidence))  # Clamp between 0-100
                
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
        return {"error": str(e)}

def save_database(db_df, file_path="face_database.pkl"):
    """Save database to file"""
    db_df.to_pickle(file_path)
    print(f"Database saved to {file_path}")

def load_database(file_path="face_database.pkl"):
    """Load database from file"""
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        print(f"Database file {file_path} not found.")
        return None

# Example usage
if __name__ == "__main__":
    # Configuration
    DATABASE_PATH = "face_database"  # Your folder with subfolders of people
    MODEL_NAME = "VGG-Face"        # Recommended model based on our tests
    DETECTOR_BACKEND = "retinaface"  # Best detector with facial landmarks
    
    # Build or load the database
    db_file = "database/face_database_{}.pkl".format(MODEL_NAME)
    if os.path.exists(db_file):
        db_df = load_database(db_file)
    else:
        db_df = build_face_database(DATABASE_PATH, MODEL_NAME, DETECTOR_BACKEND)
        save_database(db_df, db_file)
    
    # Identify a new face
    test_image = "img1.jpg"  # Replace with your test image path
    if os.path.exists(test_image):
        results = identify_face(test_image, db_df, MODEL_NAME, DETECTOR_BACKEND)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            for i, result in enumerate(results):
                print(f"\nFace #{i+1} Results:")
                print(f"  Best match: {result['best_match']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Verified: {result['verified']}")
                print(f"  Distance: {result['distance']:.4f} (threshold: {result['threshold']})")
                print(f"  Facial area: {result['facial_area']}")
    else:
        print(f"Test image {test_image} not found.")

    test_image = "img2.jpg"  # Replace with your test image path
    if os.path.exists(test_image):
        results = identify_face(test_image, db_df, MODEL_NAME, DETECTOR_BACKEND)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            for i, result in enumerate(results):
                print(f"\nFace #{i+1} Results:")
                print(f"  Best match: {result['best_match']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Verified: {result['verified']}")
                print(f"  Distance: {result['distance']:.4f} (threshold: {result['threshold']})")
                print(f"  Facial area: {result['facial_area']}")
    else:
        print(f"Test image {test_image} not found.")
