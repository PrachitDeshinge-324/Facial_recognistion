"""
Enhanced Face Database Builder with Quality Assessment
This script builds a more robust face database with quality scoring and validation.
"""

import os
import cv2 as cv
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceDatabaseBuilder:
    def __init__(self, model_name='buffalo_l', min_face_quality=0.3):
        """
        Initialize the enhanced face database builder.
        
        Args:
            model_name: InsightFace model name for better embeddings
            min_face_quality: Minimum face quality threshold to include in database
        """
        logger.info("Initializing Enhanced Face Database Builder...")
        self.app = FaceAnalysis(
            name=model_name, 
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.min_face_quality = min_face_quality
        logger.info(f"Model: {model_name}")
        logger.info(f"Minimum face quality threshold: {min_face_quality}")
        
    def calculate_face_quality(self, face, image_shape) -> float:
        """
        Calculate comprehensive face quality score.
        
        Args:
            face: Face object from insightface
            image_shape: Shape of the source image (h, w, c)
            
        Returns:
            Quality score between 0 and 1
        """
        quality_score = 0.0
        
        # Factor 1: Detection confidence
        if hasattr(face, 'det_score'):
            quality_score += face.det_score * 0.25
        
        # Factor 2: Face size relative to image
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = image_shape[0] * image_shape[1]
        size_ratio = min(face_area / image_area, 0.5) / 0.5  # Normalize to max 50% of image
        quality_score += size_ratio * 0.25
        
        # Factor 3: Face position (centered faces are often better)
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        image_center_x = image_shape[1] / 2
        image_center_y = image_shape[0] / 2
        
        center_distance = np.sqrt(
            ((face_center_x - image_center_x) / image_center_x) ** 2 +
            ((face_center_y - image_center_y) / image_center_y) ** 2
        )
        center_score = max(0, 1 - center_distance)
        quality_score += center_score * 0.15
        
        # Factor 4: Face orientation (frontal faces preferred)
        if hasattr(face, 'pose'):
            pose_angles = np.abs(face.pose)
            avg_pose_angle = np.mean(pose_angles)
            pose_score = max(0, 1 - (avg_pose_angle / 45.0))  # Penalize angles > 45 degrees
            quality_score += pose_score * 0.20
        else:
            quality_score += 0.10  # Default if pose not available
        
        # Factor 5: Age factor (working-age adults often have clearer features)
        if hasattr(face, 'age'):
            # Optimal age range for recognition: 20-50
            if 20 <= face.age <= 50:
                age_score = 1.0
            elif face.age < 20:
                age_score = max(0, face.age / 20.0)
            else:  # age > 50
                age_score = max(0, 1.0 - (face.age - 50) / 50.0)
            quality_score += age_score * 0.15
        else:
            quality_score += 0.075  # Default if age not available
        
        return min(quality_score, 1.0)
    
    def extract_enhanced_face_embedding(self, image_path: str) -> Tuple[np.ndarray, float, Dict]:
        """
        Extract face embedding with quality assessment and metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (embedding, quality_score, metadata) or (None, 0, {}) if failed
        """
        try:
            # Read the image
            img = cv.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image {image_path}")
                return None, 0.0, {}
            
            # Get faces from the image
            faces = self.app.get(img)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in {image_path}")
                return None, 0.0, {"error": "no_face_detected"}
            
            # If multiple faces, choose the best quality one
            if len(faces) > 1:
                face_qualities = []
                for face in faces:
                    quality = self.calculate_face_quality(face, img.shape)
                    face_qualities.append(quality)
                
                best_face_idx = np.argmax(face_qualities)
                best_face = faces[best_face_idx]
                logger.info(f"Multiple faces in {image_path}, selected best quality face")
            else:
                best_face = faces[0]
            
            # Calculate quality score
            quality_score = self.calculate_face_quality(best_face, img.shape)
            
            # Create metadata
            metadata = {
                "image_path": image_path,
                "face_count": len(faces),
                "quality_score": quality_score,
                "image_size": img.shape[:2],
                "face_bbox": best_face.bbox.tolist(),
                "detection_confidence": float(best_face.det_score) if hasattr(best_face, 'det_score') else None,
                "age": float(best_face.age) if hasattr(best_face, 'age') else None,
                "gender": int(best_face.gender) if hasattr(best_face, 'gender') else None,
                "pose": best_face.pose.tolist() if hasattr(best_face, 'pose') else None
            }
            
            return best_face.embedding, quality_score, metadata
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, 0.0, {"error": str(e)}
    
    def build_enhanced_database(self, database_path="face_database", output_file="enhanced_face_database.pkl"):
        """
        Build enhanced face database with quality filtering and validation.
        
        Args:
            database_path: Path to the directory containing face images
            output_file: Output file for the enhanced database
        """
        logger.info(f"Building enhanced face database from '{database_path}'...")
        
        # Initialize storage
        embeddings = []
        names = []
        qualities = []
        metadata_list = []
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics
        stats = {
            'total_images': 0,
            'images_processed': 0,
            'faces_detected': 0,
            'high_quality_faces': 0,
            'medium_quality_faces': 0,
            'low_quality_faces': 0,
            'rejected_faces': 0
        }
        
        # Process each person's folder
        database_path = Path(database_path)
        if not database_path.exists():
            logger.error(f"Database path '{database_path}' does not exist!")
            return False
        
        person_folders = [f for f in database_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(person_folders)} person folders")
        
        for person_folder in person_folders:
            person_name = person_folder.name
            logger.info(f"Processing {person_name}...")
            
            # Get all image files
            image_files = []
            for ext in supported_extensions:
                image_files.extend(person_folder.glob(f'*{ext}'))
                image_files.extend(person_folder.glob(f'*{ext.upper()}'))
            
            logger.info(f"  Found {len(image_files)} images for {person_name}")
            stats['total_images'] += len(image_files)
            
            person_embeddings = []
            person_qualities = []
            person_metadata = []
            
            # Process each image
            for image_file in image_files:
                embedding, quality, metadata = self.extract_enhanced_face_embedding(str(image_file))
                stats['images_processed'] += 1
                
                if embedding is not None:
                    stats['faces_detected'] += 1
                    
                    # Quality-based filtering
                    if quality >= self.min_face_quality:
                        person_embeddings.append(embedding)
                        person_qualities.append(quality)
                        person_metadata.append(metadata)
                        
                        # Update quality statistics
                        if quality >= 0.7:
                            stats['high_quality_faces'] += 1
                        elif quality >= 0.5:
                            stats['medium_quality_faces'] += 1
                        else:
                            stats['low_quality_faces'] += 1
                        
                        logger.info(f"    ✓ {image_file.name}: Quality {quality:.3f}")
                    else:
                        stats['rejected_faces'] += 1
                        logger.warning(f"    ✗ {image_file.name}: Quality {quality:.3f} below threshold")
                else:
                    logger.warning(f"    ✗ {image_file.name}: Failed to extract embedding")
            
            # Add person's embeddings to database
            if person_embeddings:
                embeddings.extend(person_embeddings)
                names.extend([person_name] * len(person_embeddings))
                qualities.extend(person_qualities)
                metadata_list.extend(person_metadata)
                
                avg_quality = np.mean(person_qualities)
                logger.info(f"  Added {len(person_embeddings)} embeddings for {person_name} (avg quality: {avg_quality:.3f})")
            else:
                logger.warning(f"  No valid embeddings found for {person_name}")
        
        # Create enhanced database
        if embeddings:
            enhanced_database = {
                'embeddings': np.array(embeddings),
                'names': names,
                'qualities': qualities,
                'metadata': metadata_list,
                'model_info': {
                    'model_name': 'buffalo_l',
                    'min_quality_threshold': self.min_face_quality,
                    'creation_timestamp': str(np.datetime64('now')),
                    'total_embeddings': len(embeddings),
                    'unique_people': len(set(names))
                },
                'statistics': stats
            }
            
            # Save the enhanced database
            with open(output_file, 'wb') as f:
                pickle.dump(enhanced_database, f)
            
            # Save metadata as JSON for inspection
            metadata_file = output_file.replace('.pkl', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'model_info': enhanced_database['model_info'],
                    'statistics': stats,
                    'quality_distribution': {
                        'high_quality': stats['high_quality_faces'],
                        'medium_quality': stats['medium_quality_faces'],
                        'low_quality': stats['low_quality_faces']
                    }
                }, f, indent=2)
            
            logger.info(f"\n=== Enhanced Database Build Complete ===")
            logger.info(f"Total embeddings: {len(embeddings)}")
            logger.info(f"Unique people: {len(set(names))}")
            logger.info(f"Average quality: {np.mean(qualities):.3f}")
            logger.info(f"Quality distribution:")
            logger.info(f"  High quality (≥0.7): {stats['high_quality_faces']}")
            logger.info(f"  Medium quality (0.5-0.7): {stats['medium_quality_faces']}")
            logger.info(f"  Low quality (0.3-0.5): {stats['low_quality_faces']}")
            logger.info(f"  Rejected (<0.3): {stats['rejected_faces']}")
            logger.info(f"Database saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            return True
        else:
            logger.error("No valid embeddings found! Database creation failed.")
            return False

def main():
    """Main function for building enhanced face database."""
    print("=== Enhanced Face Database Builder with ArcFace ===")
    
    # Initialize the enhanced database builder
    builder = EnhancedFaceDatabaseBuilder(
        model_name='buffalo_l',  # Use reliable model
        min_face_quality=0.3     # Minimum quality threshold
    )
    
    # Build the enhanced database
    success = builder.build_enhanced_database(
        database_path="face_database",
        output_file="enhanced_face_database.pkl"
    )
    
    if success:
        print("\n✓ Enhanced face database created successfully!")
        print("You can now use 'enhanced_face_recognition.py' for recognition.")
    else:
        print("\n✗ Failed to create enhanced face database.")

if __name__ == "__main__":
    main()
