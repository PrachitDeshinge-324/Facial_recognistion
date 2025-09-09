"""
Ultra-Robust Face Recognition System with DeepFace + InsightFace Hybrid
This script provides maximum robustness using multiple models and advanced techniques.
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
import logging
from typing import Tuple, Optional, Dict, List, Union
import warnings
warnings.filterwarnings("ignore")

# Primary engine: DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✓ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠ DeepFace not available, using InsightFace only")

# Fallback engine: InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("✓ InsightFace loaded successfully")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠ InsightFace not available")

from sklearn.metrics.pairwise import cosine_similarity
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraRobustFaceRecognitionSystem:
    def __init__(self, 
                 database_file="ultra_robust_face_database.pkl",
                 primary_model='ArcFace',  # DeepFace model
                 fallback_model='buffalo_l',  # InsightFace model
                 threshold=0.4,
                 confidence_mode='multi_model',
                 enable_anti_spoofing=True,
                 enable_quality_filter=True):
        """
        Initialize ultra-robust face recognition with multiple models.
        
        Args:
            database_file: Path to the face database
            primary_model: DeepFace model ('VGG-Face', 'Facenet', 'ArcFace', 'Dlib', 'SFace')
            fallback_model: InsightFace model for fallback
            threshold: Base similarity threshold
            confidence_mode: 'single', 'ensemble', 'multi_model'
            enable_anti_spoofing: Enable spoofing detection
            enable_quality_filter: Enable quality assessment
        """
        logger.info("🚀 Initializing Ultra-Robust Face Recognition System...")
        
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.base_threshold = threshold
        self.confidence_mode = confidence_mode
        self.enable_anti_spoofing = enable_anti_spoofing
        self.enable_quality_filter = enable_quality_filter
        
        # Initialize models
        self.deepface_models = []
        self.insightface_app = None
        
        # Setup DeepFace models
        if DEEPFACE_AVAILABLE:
            self.setup_deepface_models()
        
        # Setup InsightFace fallback
        if INSIGHTFACE_AVAILABLE:
            self.setup_insightface_fallback()
        
        # Enhanced threshold system
        self.adaptive_thresholds = {
            'high_quality': threshold + 0.15,
            'medium_quality': threshold,
            'low_quality': threshold - 0.15,
            'multi_model_consensus': threshold + 0.1
        }
        
        # Load face database
        self.face_database = None
        self.load_enhanced_database(database_file)
        
        # Performance tracking
        self.recognition_stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'anti_spoofing_rejected': 0,
            'quality_rejected': 0,
            'deepface_success': 0,
            'insightface_fallback': 0,
            'multi_model_consensus': 0
        }
        
        logger.info(f"✅ System initialized with {len(self.deepface_models)} DeepFace model(s)")
        
    def setup_deepface_models(self):
        """Setup multiple DeepFace models for ensemble recognition."""
        if not DEEPFACE_AVAILABLE:
            return
            
        # Primary models for ensemble (ordered by reliability)
        model_priority = [
            'ArcFace',      # Best accuracy
            'Facenet',      # Good speed/accuracy balance  
            'VGG-Face',     # Robust baseline
            'SFace'         # Latest OpenCV model
        ]
        
        for model in model_priority:
            try:
                # Test model availability
                DeepFace.build_model(model)
                self.deepface_models.append(model)
                logger.info(f"✓ DeepFace model '{model}' loaded")
                
                # For speed, limit to top 2 models in real-time
                if len(self.deepface_models) >= 2:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠ Could not load DeepFace model '{model}': {e}")
        
        if not self.deepface_models:
            logger.error("❌ No DeepFace models available")
    
    def setup_insightface_fallback(self):
        """Setup InsightFace as fallback system."""
        if not INSIGHTFACE_AVAILABLE:
            return
            
        try:
            self.insightface_app = FaceAnalysis(
                name=self.fallback_model,
                providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"✓ InsightFace fallback '{self.fallback_model}' ready")
        except Exception as e:
            logger.error(f"❌ InsightFace fallback failed: {e}")
            self.insightface_app = None
    
    def detect_faces_deepface(self, frame):
        """Detect faces using DeepFace with enhanced preprocessing."""
        if not DEEPFACE_AVAILABLE or not self.deepface_models:
            return []
        
        try:
            # Convert frame to proper format for DeepFace
            if isinstance(frame, np.ndarray):
                # Save frame temporarily for DeepFace processing
                temp_path = "/tmp/temp_frame.jpg"
                cv.imwrite(temp_path, frame)
                input_path = temp_path
            else:
                input_path = frame
            
            # Use DeepFace to extract faces
            result = DeepFace.extract_faces(
                img_path=input_path,
                detector_backend='opencv',
                enforce_detection=False,
                align=True,
                normalize_face=True
            )
            
            faces = []
            for i, face_region in enumerate(result):
                if face_region is not None and hasattr(face_region, 'shape'):
                    # Convert to format compatible with our system
                    face_info = {
                        'embedding': None,  # Will be computed later
                        'region': face_region,
                        'bbox': self.estimate_bbox_from_face(frame, face_region, i),
                        'quality_score': self.calculate_deepface_quality(face_region)
                    }
                    faces.append(face_info)
            
            # Clean up temp file
            if isinstance(frame, np.ndarray) and os.path.exists("/tmp/temp_frame.jpg"):
                os.remove("/tmp/temp_frame.jpg")
            
            return faces
            
        except Exception as e:
            logger.warning(f"DeepFace detection failed: {e}")
            return []
    
    def detect_faces_insightface(self, frame):
        """Fallback face detection using InsightFace."""
        if not self.insightface_app:
            return []
        
        try:
            faces = self.insightface_app.get(frame)
            return faces
        except Exception as e:
            logger.warning(f"InsightFace detection failed: {e}")
            return []
    
    def estimate_bbox_from_face(self, frame, face_region, index):
        """Estimate bounding box from face region with better detection."""
        try:
            if hasattr(face_region, 'shape'):
                h, w = frame.shape[:2]
                face_h, face_w = face_region.shape[:2]
                
                # Use a more sophisticated approach for bbox estimation
                # This is simplified - in production you'd want face detection coordinates
                scale_factor = min(h/face_h, w/face_w) * 0.8
                estimated_size = int(min(face_h, face_w) * scale_factor)
                
                # Center the face in frame
                x = max(0, (w - estimated_size) // 2)
                y = max(0, (h - estimated_size) // 2)
                
                return [x, y, min(x + estimated_size, w), min(y + estimated_size, h)]
            else:
                # Fallback bbox
                h, w = frame.shape[:2]
                return [w//4, h//4, 3*w//4, 3*h//4]
                
        except Exception as e:
            logger.warning(f"Bbox estimation failed: {e}")
            h, w = frame.shape[:2]
            return [w//4, h//4, 3*w//4, 3*h//4]
    
    def calculate_deepface_quality(self, face_region):
        """Calculate face quality score for DeepFace detected faces."""
        if face_region is None:
            return 0.0
        
        try:
            # Ensure face_region is a numpy array
            if not isinstance(face_region, np.ndarray):
                return 0.5  # Default quality if we can't process
            
            # Basic quality metrics
            quality_score = 0.0
            
            # Convert to uint8 if needed
            if face_region.dtype != np.uint8:
                if face_region.max() <= 1.0:
                    face_region = (face_region * 255).astype(np.uint8)
                else:
                    face_region = face_region.astype(np.uint8)
            
            # Convert to grayscale for analysis
            if len(face_region.shape) == 3:
                gray = cv.cvtColor(face_region, cv.COLOR_RGB2GRAY)
            else:
                gray = face_region
            
            # Factor 1: Image sharpness (Laplacian variance)
            sharpness = cv.Laplacian(gray, cv.CV_64F).var()
            normalized_sharpness = min(sharpness / 1000.0, 1.0)
            quality_score += normalized_sharpness * 0.3
            
            # Factor 2: Face size
            face_area = gray.shape[0] * gray.shape[1]
            size_score = min(face_area / (112 * 112), 1.0)  # Normalize to standard face size
            quality_score += size_score * 0.25
            
            # Factor 3: Brightness consistency
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
            quality_score += max(brightness_score, 0) * 0.25
            
            # Factor 4: Contrast
            contrast = np.std(gray) / 255.0
            contrast_score = min(contrast * 4, 1.0)  # Higher contrast usually better
            quality_score += contrast_score * 0.2
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5  # Default quality on error
    
    def get_face_embedding_deepface(self, face_region, model_name):
        """Get face embedding using specific DeepFace model."""
        try:
            # Ensure face_region is properly formatted
            if not isinstance(face_region, np.ndarray):
                logger.warning(f"Invalid face region type: {type(face_region)}")
                return None
            
            # Convert face region to proper format
            if face_region.dtype != np.uint8:
                if face_region.max() <= 1.0:
                    face_img = (face_region * 255).astype(np.uint8)
                else:
                    face_img = face_region.astype(np.uint8)
            else:
                face_img = face_region
            
            # Save temporarily for DeepFace
            temp_path = "/tmp/temp_face.jpg"
            cv.imwrite(temp_path, face_img)
            
            # Get embedding
            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=model_name,
                enforce_detection=False,
                normalize=True
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            return None
            
        except Exception as e:
            logger.warning(f"Embedding extraction failed for {model_name}: {e}")
            return None
    
    def multi_model_recognition(self, face_region):
        """Perform recognition using multiple models for consensus."""
        if not DEEPFACE_AVAILABLE or not self.deepface_models:
            return self.fallback_recognition(face_region)
        
        model_results = []
        
        # Try each DeepFace model
        for model_name in self.deepface_models:
            try:
                embedding = self.get_face_embedding_deepface(face_region, model_name)
                if embedding is not None:
                    # Compare with database
                    best_match, similarity = self.compare_with_database_deepface(embedding)
                    model_results.append({
                        'model': model_name,
                        'match': best_match,
                        'similarity': similarity,
                        'embedding': embedding
                    })
                    self.recognition_stats['deepface_success'] += 1
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        # Consensus decision
        if len(model_results) >= 2:
            return self.ensemble_consensus(model_results)
        elif len(model_results) == 1:
            result = model_results[0]
            return result['match'], result['similarity'], {'single_model': result['model']}
        else:
            # Fall back to InsightFace
            return self.fallback_recognition(face_region)
    
    def ensemble_consensus(self, model_results):
        """Make consensus decision from multiple model results."""
        if not model_results:
            return "Unknown", 0.0, {}
        
        # Voting system
        vote_counts = {}
        weighted_similarities = {}
        
        model_weights = {
            'ArcFace': 1.0,
            'Facenet': 0.9,
            'VGG-Face': 0.8,
            'SFace': 0.85
        }
        
        for result in model_results:
            name = result['match']
            similarity = result['similarity']
            model = result['model']
            weight = model_weights.get(model, 0.7)
            
            if name not in vote_counts:
                vote_counts[name] = 0
                weighted_similarities[name] = 0
            
            vote_counts[name] += weight
            weighted_similarities[name] += similarity * weight
        
        # Find consensus
        if vote_counts:
            best_name = max(vote_counts.keys(), key=lambda x: vote_counts[x])
            consensus_similarity = weighted_similarities[best_name] / vote_counts[best_name]
            
            # Require at least 60% consensus for positive identification
            consensus_threshold = len(model_results) * 0.6
            
            if vote_counts[best_name] >= consensus_threshold and consensus_similarity >= self.base_threshold:
                self.recognition_stats['multi_model_consensus'] += 1
                return best_name, consensus_similarity, {
                    'consensus_score': vote_counts[best_name] / len(model_results),
                    'models_used': [r['model'] for r in model_results]
                }
        
        return "Unknown", 0.0, {'consensus_failed': True}
    
    def compare_with_database_deepface(self, embedding):
        """Compare embedding with database using DeepFace metrics."""
        if self.face_database is None or len(self.face_database['embeddings']) == 0:
            return "Unknown", 0.0
        
        best_similarity = -1
        best_match = "Unknown"
        
        for i, db_embedding in enumerate(self.face_database['embeddings']):
            # Multiple similarity metrics
            cosine_sim = cosine_similarity([embedding], [db_embedding])[0][0]
            
            # Euclidean distance (converted to similarity)
            euclidean_dist = np.linalg.norm(embedding - db_embedding)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Combined similarity
            combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
            
            if combined_sim > best_similarity:
                best_similarity = combined_sim
                best_match = self.face_database['names'][i]
        
        return best_match, best_similarity
    
    def fallback_recognition(self, face_region):
        """Fallback to InsightFace recognition."""
        if not self.insightface_app:
            return "Unknown", 0.0, {'fallback_failed': True}
        
        try:
            # Convert face_region to frame format for InsightFace
            if isinstance(face_region, np.ndarray):
                if face_region.dtype != np.uint8:
                    if face_region.max() <= 1.0:
                        frame = (face_region * 255).astype(np.uint8)
                    else:
                        frame = face_region.astype(np.uint8)
                else:
                    frame = face_region
                
                # Ensure proper color format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            else:
                logger.warning(f"Invalid face region for fallback: {type(face_region)}")
                return "Unknown", 0.0, {'fallback_failed': True}
            
            faces = self.insightface_app.get(frame)
            
            if faces and self.face_database and 'embeddings' in self.face_database:
                face = faces[0]  # Use first detected face
                embedding = face.embedding
                
                # Use your existing recognition logic
                similarities = cosine_similarity([embedding], self.face_database['embeddings'])[0]
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity >= self.base_threshold:
                    recognized_name = self.face_database['names'][best_match_idx]
                    self.recognition_stats['insightface_fallback'] += 1
                    return recognized_name, best_similarity, {'fallback_used': 'InsightFace'}
                
            return "Unknown", 0.0, {'fallback_used': 'InsightFace'}
            
        except Exception as e:
            logger.error(f"Fallback recognition failed: {e}")
            return "Unknown", 0.0, {'fallback_failed': True}
    
    def load_enhanced_database(self, database_file):
        """Load the enhanced face database."""
        if not os.path.exists(database_file):
            logger.warning(f"Database '{database_file}' not found. Trying alternative...")
            # Try other database files
            alternatives = ["enhanced_face_database.pkl", "face_database.pkl"]
            for alt_file in alternatives:
                if os.path.exists(alt_file):
                    database_file = alt_file
                    logger.info(f"Using alternative database: {alt_file}")
                    break
            else:
                logger.error("No face database found!")
                return False
        
        try:
            with open(database_file, 'rb') as f:
                self.face_database = pickle.load(f)
            
            logger.info(f"✅ Face database loaded: {len(self.face_database['embeddings'])} embeddings")
            logger.info(f"Unique people: {len(set(self.face_database['names']))}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame with ultra-robust recognition."""
        results = []
        
        # Primary detection with DeepFace
        deepface_faces = self.detect_faces_deepface(frame)
        
        if deepface_faces:
            for face_info in deepface_faces:
                # Quality filtering
                if self.enable_quality_filter and face_info['quality_score'] < 0.3:
                    self.recognition_stats['quality_rejected'] += 1
                    continue
                
                # Multi-model recognition
                name, confidence, details = self.multi_model_recognition(face_info['region'])
                
                # Anti-spoofing check (simplified)
                if self.enable_anti_spoofing and self.is_spoofing_detected(face_info['region']):
                    self.recognition_stats['anti_spoofing_rejected'] += 1
                    continue
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'bbox': face_info['bbox'],
                    'quality': face_info['quality_score'],
                    'details': details
                })
        
        else:
            # Fallback to InsightFace
            insightface_faces = self.detect_faces_insightface(frame)
            for face in insightface_faces:
                name, confidence, details = self.fallback_recognition(frame)
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'bbox': face.bbox.astype(int),
                    'quality': 0.5,  # Default quality
                    'details': details
                })
        
        # Update statistics
        self.recognition_stats['total_faces'] += len(results)
        for result in results:
            if result['name'] != "Unknown":
                self.recognition_stats['recognized_faces'] += 1
                if result['confidence'] > self.base_threshold + 0.2:
                    self.recognition_stats['high_confidence'] += 1
                elif result['confidence'] > self.base_threshold + 0.1:
                    self.recognition_stats['medium_confidence'] += 1
                else:
                    self.recognition_stats['low_confidence'] += 1
            else:
                self.recognition_stats['unknown_faces'] += 1
        
        return results
    
    def is_spoofing_detected(self, face_region):
        """Simple anti-spoofing detection (placeholder for advanced methods)."""
        # This is a simplified version - in production, use specialized models
        try:
            # Check for image texture patterns that indicate printed photos
            gray = cv.cvtColor((face_region * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            
            # Calculate local binary patterns or other texture features
            # For now, just check if the image looks too uniform (could be a print)
            texture_variance = np.var(gray)
            
            # Very low texture variance might indicate a printed photo
            if texture_variance < 100:  # Threshold needs tuning
                return True
            
            return False
            
        except Exception:
            return False  # If analysis fails, don't reject
    
    def get_recognition_stats(self):
        """Get comprehensive recognition statistics."""
        stats = self.recognition_stats.copy()
        total = stats['total_faces']
        
        if total > 0:
            stats['recognition_rate'] = stats['recognized_faces'] / total
            stats['unknown_rate'] = stats['unknown_faces'] / total
            stats['quality_rejection_rate'] = stats['quality_rejected'] / (total + stats['quality_rejected'])
            stats['anti_spoofing_rejection_rate'] = stats['anti_spoofing_rejected'] / (total + stats['anti_spoofing_rejected'])
        
        return stats

def main():
    """Main function for ultra-robust face recognition."""
    print("🚀 === Ultra-Robust Face Recognition System ===")
    print("📊 Features: DeepFace + InsightFace | Multi-Model Ensemble | Anti-Spoofing")
    
    # Initialize ultra-robust system
    try:
        recognition_system = UltraRobustFaceRecognitionSystem(
            primary_model='ArcFace',
            fallback_model='buffalo_l',
            threshold=0.4,
            confidence_mode='multi_model',
            enable_anti_spoofing=True,
            enable_quality_filter=True
        )
        
        # Video source
        # video_path = 0  # Webcam
        video_path = '../Facial Recognision/video/03_09_2025_face_recognition.mp4'
        
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video source")
            return
        
        print("🎥 Starting ultra-robust recognition...")
        print("📝 Press 'q' to quit, 's' for statistics, 'r' for recognition report")
        
        frame_count = 0
        total_fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            frame_count += 1
            
            # Process frame with ultra-robust recognition
            face_results = recognition_system.process_frame(frame)
            
            # Draw results
            for result in face_results:
                bbox = result['bbox']
                name = result['name']
                confidence = result['confidence']
                quality = result['quality']
                
                # Color coding based on confidence and method
                if 'multi_model_consensus' in str(result['details']):
                    color = (0, 255, 0)  # Green for consensus
                elif 'fallback_used' in result['details']:
                    color = (0, 165, 255)  # Orange for fallback
                elif name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (255, 0, 255)  # Magenta for single model
                
                # Draw bounding box
                cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw information
                info_text = f"{name} ({confidence:.3f})"
                quality_text = f"Q:{quality:.2f}"
                
                cv.putText(frame, info_text, (bbox[0], bbox[1]-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv.putText(frame, quality_text, (bbox[0], bbox[3]+20), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - start_time)
            total_fps += fps
            avg_fps = total_fps / frame_count
            
            cv.putText(frame, f"FPS: {fps:.1f} | Avg: {avg_fps:.1f}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv.putText(frame, f"Faces: {len(face_results)} | Ultra-Robust Mode", 
                      (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv.imshow('Ultra-Robust Face Recognition', frame)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = recognition_system.get_recognition_stats()
                print("\n📊 Current Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif key == ord('r'):
                print("\n📈 Recognition Report:")
                stats = recognition_system.get_recognition_stats()
                print(f"  Recognition Rate: {stats.get('recognition_rate', 0):.2%}")
                print(f"  Multi-Model Consensus: {stats.get('multi_model_consensus', 0)}")
                print(f"  DeepFace Success: {stats.get('deepface_success', 0)}")
                print(f"  InsightFace Fallback: {stats.get('insightface_fallback', 0)}")
        
        # Final cleanup and statistics
        cap.release()
        cv.destroyAllWindows()
        
        final_stats = recognition_system.get_recognition_stats()
        print("\n🎯 === Final Ultra-Robust Recognition Report ===")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Ultra-robust face recognition completed!")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()
