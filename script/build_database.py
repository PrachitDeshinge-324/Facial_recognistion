"""Build face recognition database from images."""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.core.detection import FaceDetector
from src.core.database import FaceDatabase
from src.config.paths import Database_Data_Path


def build_database(visualize=False):
    """Build face database from image directory."""
    
    print("=== Face Database Builder ===")
    
    # Initialize detector and database
    detector = FaceDetector()
    database = FaceDatabase()
    
    database_dir = Database_Data_Path
    if not database_dir.exists():
        print(f"Error: Database path {database_dir} does not exist")
        return False
        
    print(f"Building face database from {database_dir.resolve()}")
    
    names = []
    embeddings = []
    landmarks = []
    image_paths = []
    failed_images = 0
    
    # Process each person directory
    for person_folder in sorted(database_dir.iterdir()):
        if not person_folder.is_dir():
            continue
            
        person_name = person_folder.name
        print(f"\nProcessing person: {person_name}")
        person_image_count = 0
        
        # Process each image
        for image_file in sorted(person_folder.iterdir()):
            if image_file.suffix.lower() not in FaceDatabase.SUPPORTED_EXTENSIONS:
                continue
                
            print(f"  Processing: {image_file.name}")
            embedding, landmarks_3d = detector.extract_single_face(image_file)
            if embedding is None:
                print(f"    Failed to extract embedding from {image_file.name}")
                failed_images += 1
                continue
                
            names.append(person_name)
            embeddings.append(embedding.astype(np.float32))
            landmarks.append(landmarks_3d)
            image_paths.append(str(image_file))
            person_image_count += 1
            
        if person_image_count:
            print(f"  Successfully processed {person_image_count} images for {person_name}")
        else:
            print(f"  Warning: No valid embeddings found for {person_name}")
    
    if not embeddings:
        print("Error: No valid embeddings found in the database")
        return False
        
    # Convert to array
    try:
        embedding_array = np.stack(embeddings)
    except ValueError:
        embedding_array = np.array(embeddings, dtype=np.float32)
        
    # Create database dictionary
    face_database = {
        "names": names,
        "embeddings": embedding_array,
        "landmarks_3d": landmarks,
        "image_paths": image_paths,
    }
    
    # Save database and metadata
    database.save(face_database)
    database.create_metadata(names, failed_images, database_dir)
    
    if visualize:
        try:
            from src.utils.visualization import visualize_embeddings
            visualize_embeddings(face_database)
        except ImportError as e:
            print(f"Visualization skipped: {e}")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build face recognition database")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize embeddings after building")
    args = parser.parse_args()
    
    success = build_database(visualize=args.visualize)
    
    if success:
        print("\n✅ Face database built successfully!")
        print("You can now run the recognition system.")
    else:
        print("\n❌ Failed to build face database")
        print("Please ensure you have valid images in the person folders.")


if __name__ == "__main__":
    main()