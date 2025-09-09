"""
Enhanced Face Recognition System with ArcFace Model
This script provides a more robust face recognition system using state-of-the-art models
with improved similarity metrics and confidence scoring.
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Tuple, Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceRecognitionSystem:
    def __init__(self, 
                 database_file="face_database.pkl", 
                 threshold=0.4,
                 model_name='buffalo_l',  # Use reliable model
                 det_size=(640, 640),
                 confidence_mode='adaptive'):
        """
        Initialize the enhanced face recognition system with ArcFace model.
        
        Args:
            database_file: Path to the pickled face database
            threshold: Base cosine similarity threshold for recognition
            model_name: InsightFace model name ('antelopev2', 'buffalo_l', etc.)
            det_size: Detection size for face detection
            confidence_mode: 'fixed', 'adaptive', or 'ensemble'
        """
        logger.info("Initializing Enhanced Face Recognition System with ArcFace...")
        
        # Initialize the face analysis model with better architecture
        self.app = FaceAnalysis(
            name=model_name, 
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # Enhanced threshold system
        self.base_threshold = threshold
        self.confidence_mode = confidence_mode
        self.adaptive_thresholds = {
            'high_quality': threshold + 0.1,
            'medium_quality': threshold,
            'low_quality': threshold - 0.1
        }
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Base recognition threshold: {threshold}")
        logger.info(f"Confidence mode: {confidence_mode}")
        
        # Load the face database
        self.face_database = None
        self.face_quality_scores = []
        self.load_face_database(database_file)
        
        # Performance tracking
        self.recognition_stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
        
    def calculate_face_quality(self, face) -> float:
        """
        Calculate face quality score based on multiple factors.
        
        Args:
            face: Face object from insightface
            
        Returns:
            Quality score between 0 and 1
        """
        quality_score = 0.0
        
        # Factor 1: Detection confidence
        if hasattr(face, 'det_score'):
            quality_score += face.det_score * 0.3
        
        # Factor 2: Face size (larger faces generally better quality)
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        normalized_area = min(face_area / (640 * 640), 1.0)  # Normalize to detection size
        quality_score += normalized_area * 0.3
        
        # Factor 3: Pose estimation (frontal faces are better)
        if hasattr(face, 'pose'):
            # Lower pose angles = better quality
            pose_penalty = np.mean(np.abs(face.pose)) / 90.0  # Normalize to 90 degrees
            quality_score += (1.0 - pose_penalty) * 0.2
        else:
            quality_score += 0.1  # Default if pose not available
        
        # Factor 4: Age estimation (middle-aged faces often have better features)
        if hasattr(face, 'age'):
            age_score = 1.0 - abs(face.age - 35) / 35.0  # Peak at age 35
            quality_score += max(age_score, 0.0) * 0.2
        else:
            quality_score += 0.1  # Default if age not available
        
        return min(quality_score, 1.0)
    
    def get_adaptive_threshold(self, face_quality: float) -> float:
        """
        Get adaptive threshold based on face quality.
        
        Args:
            face_quality: Quality score of the face
            
        Returns:
            Adjusted threshold
        """
        if face_quality > 0.7:
            return self.adaptive_thresholds['high_quality']
        elif face_quality > 0.4:
            return self.adaptive_thresholds['medium_quality']
        else:
            return self.adaptive_thresholds['low_quality']
    
    def arcface_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate ArcFace-style similarity with angular margin.
        
        Args:
            embedding1, embedding2: Face embeddings
            
        Returns:
            Enhanced similarity score
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1_norm, embedding2_norm)
        
        # Apply angular margin enhancement (ArcFace concept)
        angular_margin = 0.5  # Angular margin parameter
        enhanced_similarity = cosine_sim - angular_margin
        
        # Ensure similarity is in valid range
        return max(min(enhanced_similarity, 1.0), -1.0)
    
    def ensemble_recognition(self, face_embedding: np.ndarray, face_quality: float) -> Tuple[str, float, Dict]:
        """
        Enhanced recognition using ensemble of similarity metrics.
        
        Args:
            face_embedding: The face embedding
            face_quality: Quality score of the face
            
        Returns:
            tuple: (recognized_name, confidence_score, detailed_info)
        """
        if self.face_database is None or len(self.face_database['embeddings']) == 0:
            return "Unknown", 0.0, {"reason": "No database"}
        
        # Calculate multiple similarity metrics
        cosine_similarities = cosine_similarity([face_embedding], self.face_database['embeddings'])[0]
        
        # Calculate ArcFace-style similarities
        arcface_similarities = []
        for db_embedding in self.face_database['embeddings']:
            arcface_sim = self.arcface_similarity(face_embedding, db_embedding)
            arcface_similarities.append(arcface_sim)
        arcface_similarities = np.array(arcface_similarities)
        
        # Ensemble scoring (weighted combination)
        ensemble_scores = 0.6 * cosine_similarities + 0.4 * arcface_similarities
        
        # Find best matches
        best_match_idx = np.argmax(ensemble_scores)
        best_score = ensemble_scores[best_match_idx]
        cosine_score = cosine_similarities[best_match_idx]
        arcface_score = arcface_similarities[best_match_idx]
        
        # Get adaptive threshold
        if self.confidence_mode == 'adaptive':
            threshold = self.get_adaptive_threshold(face_quality)
        else:
            threshold = self.base_threshold
        
        # Detailed information
        detailed_info = {
            "cosine_similarity": float(cosine_score),
            "arcface_similarity": float(arcface_score),
            "ensemble_score": float(best_score),
            "threshold_used": float(threshold),
            "face_quality": float(face_quality),
            "top_3_matches": []
        }
        
        # Get top 3 matches for analysis
        top_3_indices = np.argsort(ensemble_scores)[-3:][::-1]
        for i, idx in enumerate(top_3_indices):
            detailed_info["top_3_matches"].append({
                "rank": i + 1,
                "name": self.face_database['names'][idx],
                "score": float(ensemble_scores[idx])
            })
        
        # Decision making
        if best_score >= threshold:
            recognized_name = self.face_database['names'][best_match_idx]
            
            # Update statistics
            self.recognition_stats['recognized_faces'] += 1
            if best_score > threshold + 0.2:
                self.recognition_stats['high_confidence'] += 1
            elif best_score > threshold + 0.1:
                self.recognition_stats['medium_confidence'] += 1
            else:
                self.recognition_stats['low_confidence'] += 1
                
            return recognized_name, best_score, detailed_info
        else:
            self.recognition_stats['unknown_faces'] += 1
            return "Unknown", best_score, detailed_info
    
    def load_face_database(self, database_file):
        """Load the pre-built face database with quality assessment."""
        if not os.path.exists(database_file):
            logger.warning(f"Face database file '{database_file}' not found!")
            logger.warning("Please run 'build_face_database.py' first to create the database.")
            logger.warning("Running in detection-only mode (no recognition).")
            return False
        
        try:
            with open(database_file, 'rb') as f:
                self.face_database = pickle.load(f)
            
            logger.info(f"Face database loaded successfully!")
            logger.info(f"Database contains {len(self.face_database['embeddings'])} embeddings")
            logger.info(f"Unique people: {len(set(self.face_database['names']))}")
            
            # Print database summary
            name_counts = {}
            for name in self.face_database['names']:
                name_counts[name] = name_counts.get(name, 0) + 1
            
            for name, count in name_counts.items():
                logger.info(f"  {name}: {count} embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
            logger.warning("Running in detection-only mode.")
            return False
    
    def draw_enhanced_face_info(self, frame, face, name, confidence, detailed_info):
        """
        Draw enhanced bounding box and recognition info on the frame.
        
        Args:
            frame: The video frame
            face: Face object from insightface
            name: Recognized name or "Unknown"
            confidence: Confidence score
            detailed_info: Detailed recognition information
        """
        # Get bounding box coordinates
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Calculate face quality
        face_quality = self.calculate_face_quality(face)
        
        # Choose colors based on recognition status and confidence
        if name == "Unknown":
            box_color = (0, 0, 255)  # Red for unknown
            text_color = (255, 255, 255)
        else:
            if confidence > self.base_threshold + 0.2:
                box_color = (0, 255, 0)  # Green for high confidence
            elif confidence > self.base_threshold + 0.1:
                box_color = (0, 255, 255)  # Yellow for medium confidence
            else:
                box_color = (0, 165, 255)  # Orange for low confidence
            text_color = (255, 255, 255)
        
        # Draw bounding box with thickness based on confidence
        thickness = 3 if confidence > self.base_threshold + 0.1 else 2
        cv.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        # Prepare enhanced text information
        texts = [
            f"{name}",
            f"Conf: {confidence:.3f}",
            f"Quality: {face_quality:.2f}",
            f"ArcFace: {detailed_info.get('arcface_similarity', 0):.3f}"
        ]
        
        # Draw text with background
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        y_offset = y1 - 10
        for i, text in enumerate(texts):
            text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y_offset - (len(texts) - 1 - i) * 20
            
            if text_y < 0:
                text_y = y2 + 20 + i * 20
            
            # Draw text background
            cv.rectangle(frame, 
                        (text_x, text_y - text_size[1] - 3), 
                        (text_x + text_size[0] + 6, text_y + 3), 
                        box_color, -1)
            
            # Draw text
            cv.putText(frame, text, (text_x + 3, text_y - 1), 
                      font, font_scale, text_color, font_thickness)
        
        # Draw quality indicator (small circle in top-right of bbox)
        quality_color = (0, 255, 0) if face_quality > 0.7 else (0, 255, 255) if face_quality > 0.4 else (0, 0, 255)
        cv.circle(frame, (x2 - 10, y1 + 10), 5, quality_color, -1)
    
    def get_recognition_stats(self) -> Dict:
        """Get current recognition statistics."""
        total = self.recognition_stats['total_faces']
        if total == 0:
            return self.recognition_stats
        
        stats = self.recognition_stats.copy()
        stats['recognition_rate'] = stats['recognized_faces'] / total
        stats['unknown_rate'] = stats['unknown_faces'] / total
        
        return stats

def process_video_enhanced(cap, recognition_system):
    """Process the video with enhanced face recognition."""
    frame_count = 0
    total_fps = 0
    
    logger.info("\nStarting enhanced face recognition...")
    logger.info("Press 'q' to quit, 's' to show statistics")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing for this frame
        start_time = time.time()
        frame_count += 1
        
        # Detect faces in the frame
        faces = recognition_system.app.get(frame)
        recognition_system.recognition_stats['total_faces'] += len(faces)
        
        # Process each detected face
        for face in faces:
            # Calculate face quality
            face_quality = recognition_system.calculate_face_quality(face)
            
            # Get the face embedding
            face_embedding = face.embedding
            
            # Enhanced recognition
            name, confidence, detailed_info = recognition_system.ensemble_recognition(
                face_embedding, face_quality
            )
            
            # Draw enhanced face info on the frame
            recognition_system.draw_enhanced_face_info(
                frame, face, name, confidence, detailed_info
            )
        
        # Calculate FPS
        current_time = time.time()
        processing_time = current_time - start_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        
        total_fps += current_fps
        average_fps = total_fps / frame_count
        
        # Display enhanced system info
        cv.putText(frame, f"FPS: {current_fps:.1f} | Avg: {average_fps:.1f}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(frame, f"Faces: {len(faces)} | Model: ArcFace Enhanced", 
                  (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(frame, f"Mode: {recognition_system.confidence_mode} | Threshold: {recognition_system.base_threshold}", 
                  (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show the frame
        cv.imshow('Enhanced Face Recognition System', frame)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = recognition_system.get_recognition_stats()
            logger.info(f"Recognition Statistics: {stats}")

def main():
    """Main function for the enhanced face recognition system."""
    print("=== Enhanced Face Recognition System with ArcFace ===")
    
    # Initialize the enhanced recognition system
    recognition_system = EnhancedFaceRecognitionSystem(
        database_file="enhanced_face_database.pkl",  # Use enhanced database
        threshold=0.4,
        model_name='buffalo_l',  # Use reliable model
        confidence_mode='adaptive'
    )
    
    # Video path
    video_path = '../Facial Recognision/video/03_09_2025_face_recognition.mp4'
    # video_path = '../Person Identification/v_1/input/3c.mp4'
    # video_path = 0  # For webcam
    
    # Open the video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video source")
        return
    
    try:
        # Process the video with enhanced recognition
        process_video_enhanced(cap, recognition_system)
    finally:
        # Cleanup and show final statistics
        cap.release()
        cv.destroyAllWindows()
        
        final_stats = recognition_system.get_recognition_stats()
        logger.info("Final Recognition Statistics:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")
        
        print("\nEnhanced face recognition system stopped.")

if __name__ == "__main__":
    main()
