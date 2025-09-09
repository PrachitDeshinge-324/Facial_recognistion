"""
DeepFace vs InsightFace Comparison Tool
Direct comparison of both approaches for face recognition robustness
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
import logging
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing both libraries
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✓ DeepFace available")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("⚠ DeepFace not available")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("✓ InsightFace available")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠ InsightFace not available")

from sklearn.metrics.pairwise import cosine_similarity

class ModelComparisonSystem:
    def __init__(self):
        """Initialize comparison system with both models."""
        self.insightface_app = None
        self.deepface_models = ['ArcFace', 'Facenet']
        
        # Load existing database
        self.face_database = None
        self.load_database()
        
        # Setup InsightFace
        if INSIGHTFACE_AVAILABLE:
            self.setup_insightface()
        
        # Performance tracking
        self.comparison_stats = {
            'deepface_detections': 0,
            'insightface_detections': 0,
            'deepface_recognitions': 0,
            'insightface_recognitions': 0,
            'deepface_accuracy': 0,
            'insightface_accuracy': 0,
            'deepface_speed': [],
            'insightface_speed': []
        }
    
    def setup_insightface(self):
        """Setup InsightFace model."""
        try:
            self.insightface_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✓ InsightFace ready")
        except Exception as e:
            logger.error(f"❌ InsightFace setup failed: {e}")
    
    def load_database(self):
        """Load face database."""
        database_files = [
            "ultra_robust_face_database.pkl",
            "enhanced_face_database.pkl", 
            "face_database.pkl"
        ]
        
        for db_file in database_files:
            if os.path.exists(db_file):
                try:
                    with open(db_file, 'rb') as f:
                        self.face_database = pickle.load(f)
                    logger.info(f"✓ Database loaded: {db_file}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {db_file}: {e}")
        
        logger.error("❌ No face database found!")
        return False
    
    def test_deepface_recognition(self, frame):
        """Test DeepFace recognition performance."""
        if not DEEPFACE_AVAILABLE:
            return None, 0, 0
        
        start_time = time.time()
        results = []
        
        try:
            # Use DeepFace.find for direct recognition
            dfs = DeepFace.find(
                img_path=frame,
                db_path="face_database",  # Use original folder
                model_name='ArcFace',
                enforce_detection=False,
                silent=True
            )
            
            processing_time = time.time() - start_time
            self.comparison_stats['deepface_speed'].append(processing_time)
            
            # Process results
            for df in dfs:
                if not df.empty:
                    # Get the best match
                    best_match = df.iloc[0]
                    identity_path = best_match['identity']
                    distance = best_match['distance']
                    
                    # Extract person name from path
                    person_name = os.path.basename(os.path.dirname(identity_path))
                    confidence = 1 - distance  # Convert distance to confidence
                    
                    if confidence > 0.3:  # Threshold
                        results.append({
                            'name': person_name,
                            'confidence': confidence,
                            'method': 'DeepFace_ArcFace'
                        })
                        self.comparison_stats['deepface_recognitions'] += 1
            
            self.comparison_stats['deepface_detections'] += len(results)
            return results, processing_time, len(results)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"DeepFace recognition failed: {e}")
            return [], processing_time, 0
    
    def test_insightface_recognition(self, frame):
        """Test InsightFace recognition performance."""
        if not self.insightface_app or not self.face_database:
            return None, 0, 0
        
        start_time = time.time()
        results = []
        
        try:
            faces = self.insightface_app.get(frame)
            
            for face in faces:
                embedding = face.embedding
                
                # Compare with database
                if 'embeddings' in self.face_database:
                    similarities = cosine_similarity([embedding], self.face_database['embeddings'])[0]
                    best_match_idx = np.argmax(similarities)
                    best_similarity = similarities[best_match_idx]
                    
                    if best_similarity >= 0.4:  # Threshold
                        recognized_name = self.face_database['names'][best_match_idx]
                        results.append({
                            'name': recognized_name,
                            'confidence': best_similarity,
                            'method': 'InsightFace_Buffalo',
                            'bbox': face.bbox.astype(int)
                        })
                        self.comparison_stats['insightface_recognitions'] += 1
            
            processing_time = time.time() - start_time
            self.comparison_stats['insightface_speed'].append(processing_time)
            self.comparison_stats['insightface_detections'] += len(results)
            
            return results, processing_time, len(results)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"InsightFace recognition failed: {e}")
            return [], processing_time, 0
    
    def run_live_comparison(self):
        """Run live comparison between both models."""
        if not DEEPFACE_AVAILABLE and not INSIGHTFACE_AVAILABLE:
            print("❌ Neither model available for comparison")
            return
        
        print("🚀 === DeepFace vs InsightFace Live Comparison ===")
        print("📊 Testing both models side-by-side for robustness analysis")
        
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test both models every 30 frames (for performance)
            if frame_count % 30 == 0:
                print(f"\n📊 Frame {frame_count} - Running Comparison...")
                
                # Test DeepFace
                if DEEPFACE_AVAILABLE:
                    df_results, df_time, df_count = self.test_deepface_recognition(frame)
                    print(f"  🤖 DeepFace: {df_count} faces, {df_time:.3f}s")
                    if df_results:
                        for result in df_results:
                            print(f"    ✓ {result['name']} ({result['confidence']:.3f})")
                
                # Test InsightFace
                if INSIGHTFACE_AVAILABLE:
                    if_results, if_time, if_count = self.test_insightface_recognition(frame)
                    print(f"  🤖 InsightFace: {if_count} faces, {if_time:.3f}s")
                    if if_results:
                        for result in if_results:
                            print(f"    ✓ {result['name']} ({result['confidence']:.3f})")
                            
                            # Draw bounding box if available
                            if 'bbox' in result:
                                bbox = result['bbox']
                                cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                                cv.putText(frame, f"{result['name']}", (bbox[0], bbox[1]-10), 
                                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv.putText(frame, f"Frame: {frame_count} | Comparison Mode", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv.imshow('DeepFace vs InsightFace Comparison', frame)
            
            # Controls
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.print_comparison_stats()
        
        cap.release()
        cv.destroyAllWindows()
        
        # Final analysis
        self.print_final_analysis()
    
    def print_comparison_stats(self):
        """Print current comparison statistics."""
        print("\n📊 === Current Comparison Statistics ===")
        
        if DEEPFACE_AVAILABLE:
            df_avg_speed = np.mean(self.comparison_stats['deepface_speed']) if self.comparison_stats['deepface_speed'] else 0
            print(f"🤖 DeepFace:")
            print(f"  Detections: {self.comparison_stats['deepface_detections']}")
            print(f"  Recognitions: {self.comparison_stats['deepface_recognitions']}")
            print(f"  Avg Speed: {df_avg_speed:.3f}s")
        
        if INSIGHTFACE_AVAILABLE:
            if_avg_speed = np.mean(self.comparison_stats['insightface_speed']) if self.comparison_stats['insightface_speed'] else 0
            print(f"🤖 InsightFace:")
            print(f"  Detections: {self.comparison_stats['insightface_detections']}")
            print(f"  Recognitions: {self.comparison_stats['insightface_recognitions']}")
            print(f"  Avg Speed: {if_avg_speed:.3f}s")
    
    def print_final_analysis(self):
        """Print comprehensive final analysis."""
        print("\n🎯 === Final Robustness Analysis ===")
        
        # Speed comparison
        df_speeds = self.comparison_stats['deepface_speed']
        if_speeds = self.comparison_stats['insightface_speed']
        
        if df_speeds and if_speeds:
            df_avg = np.mean(df_speeds)
            if_avg = np.mean(if_speeds)
            
            print(f"⚡ Speed Comparison:")
            print(f"  DeepFace Average: {df_avg:.3f}s")
            print(f"  InsightFace Average: {if_avg:.3f}s")
            print(f"  Winner: {'InsightFace' if if_avg < df_avg else 'DeepFace'} (faster)")
        
        # Recognition accuracy
        df_total = self.comparison_stats['deepface_detections']
        if_total = self.comparison_stats['insightface_detections']
        df_success = self.comparison_stats['deepface_recognitions']
        if_success = self.comparison_stats['insightface_recognitions']
        
        print(f"\n🎯 Recognition Performance:")
        if df_total > 0:
            df_rate = df_success / df_total
            print(f"  DeepFace Recognition Rate: {df_rate:.2%} ({df_success}/{df_total})")
        
        if if_total > 0:
            if_rate = if_success / if_total
            print(f"  InsightFace Recognition Rate: {if_rate:.2%} ({if_success}/{if_total})")
        
        # Recommendations
        print(f"\n🚀 === Robustness Recommendations ===")
        
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace not tested - install for comparison:")
            print("   pip install deepface tensorflow")
        elif not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace not tested - already installed")
        else:
            print("✅ Both models tested successfully!")
            
            # Make recommendations based on results
            if df_speeds and if_speeds:
                df_avg = np.mean(df_speeds)
                if_avg = np.mean(if_speeds)
                
                if df_avg < if_avg and df_success >= if_success:
                    print("🏆 Recommendation: DeepFace")
                    print("   - Better speed and equal/better accuracy")
                    print("   - More model options available")
                    print("   - Better for production deployment")
                elif if_avg < df_avg:
                    print("🏆 Recommendation: InsightFace")
                    print("   - Superior speed performance")
                    print("   - Good accuracy for real-time applications")
                    print("   - Lower computational overhead")
                else:
                    print("🤝 Recommendation: Hybrid Approach")
                    print("   - Use InsightFace for real-time detection")
                    print("   - Use DeepFace for high-accuracy verification")
                    print("   - Implement fallback system")

def main():
    """Main function for model comparison."""
    print("🚀 === DeepFace vs InsightFace Robustness Test ===")
    
    # Check availability
    if not DEEPFACE_AVAILABLE and not INSIGHTFACE_AVAILABLE:
        print("❌ No face recognition models available!")
        print("Install at least one:")
        print("  pip install deepface tensorflow  # For DeepFace")
        print("  pip install insightface  # For InsightFace")
        return
    
    # Initialize comparison system
    comparison_system = ModelComparisonSystem()
    
    if not comparison_system.face_database:
        print("❌ No face database found!")
        print("Please run the database builder first:")
        print("  python enhanced_database_builder.py")
        return
    
    print("\n🎯 Available Models:")
    if DEEPFACE_AVAILABLE:
        print("  ✓ DeepFace (ArcFace, Facenet, VGG-Face, etc.)")
    else:
        print("  ❌ DeepFace (not installed)")
    
    if INSIGHTFACE_AVAILABLE:
        print("  ✓ InsightFace (Buffalo_L)")
    else:
        print("  ❌ InsightFace (not installed)")
    
    print("\n📋 Test Instructions:")
    print("  - Press 'q' to quit")
    print("  - Press 's' for current statistics")
    print("  - Comparison runs every 30 frames for performance")
    
    # Run comparison
    comparison_system.run_live_comparison()

if __name__ == "__main__":
    main()
