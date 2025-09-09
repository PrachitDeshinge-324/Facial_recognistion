"""
Ultra-Robust Face Database Builder
Creates multi-model face embeddings for maximum robustness
"""

import os
import cv2 as cv
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✓ DeepFace available for ultra-robust database building")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("⚠ DeepFace not available - falling back to InsightFace only")

# Try importing InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("✓ InsightFace available for fallback embeddings")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠ InsightFace not available")

from sklearn.metrics.pairwise import cosine_similarity

class UltraRobustDatabaseBuilder:
    def __init__(self, 
                 face_database_folder="face_database", 
                 output_file="ultra_robust_face_database.pkl",
                 primary_models=['ArcFace', 'Facenet'],
                 fallback_model='buffalo_l',
                 min_face_size=80,
                 quality_threshold=0.3):
        """
        Initialize ultra-robust database builder.
        
        Args:
            face_database_folder: Folder containing face images
            output_file: Output database file
            primary_models: List of DeepFace models to use
            fallback_model: InsightFace model for fallback
            min_face_size: Minimum face size for processing
            quality_threshold: Minimum quality score for inclusion
        """
        self.face_database_folder = face_database_folder
        self.output_file = output_file
        self.primary_models = primary_models
        self.fallback_model = fallback_model
        self.min_face_size = min_face_size
        self.quality_threshold = quality_threshold
        
        # Initialize models
        self.available_deepface_models = []
        self.insightface_app = None
        
        # Setup models
        self.setup_models()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'faces_detected': 0,
            'quality_passed': 0,
            'embeddings_created': 0,
            'people_processed': 0,
            'deepface_embeddings': 0,
            'insightface_embeddings': 0,
            'multi_model_embeddings': 0
        }
    
    def setup_models(self):
        """Setup both DeepFace and InsightFace models."""
        # Setup DeepFace models
        if DEEPFACE_AVAILABLE:
            for model in self.primary_models:
                try:
                    # Test model availability
                    DeepFace.build_model(model)
                    self.available_deepface_models.append(model)
                    logger.info(f"✓ DeepFace model '{model}' ready for database building")
                except Exception as e:
                    logger.warning(f"⚠ DeepFace model '{model}' failed: {e}")
        
        # Setup InsightFace fallback
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface_app = FaceAnalysis(
                    name=self.fallback_model,
                    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
                )
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info(f"✓ InsightFace model '{self.fallback_model}' ready for fallback")
            except Exception as e:
                logger.error(f"❌ InsightFace setup failed: {e}")
    
    def calculate_image_quality(self, image):
        """Calculate comprehensive image quality score."""
        quality_score = 0.0
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Factor 1: Sharpness (Laplacian variance) - 30%
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        quality_score += sharpness_score * 0.30
        
        # Factor 2: Brightness distribution - 25%
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        quality_score += brightness_score * 0.25
        
        # Factor 3: Contrast (standard deviation) - 25%
        contrast = np.std(gray)
        contrast_score = min(contrast / 64.0, 1.0)
        quality_score += contrast_score * 0.25
        
        # Factor 4: Image size adequacy - 20%
        height, width = gray.shape
        min_dim = min(height, width)
        size_score = min(min_dim / self.min_face_size, 1.0)
        quality_score += size_score * 0.20
        
        return min(quality_score, 1.0)
    
    def extract_faces_deepface(self, image_path, person_name):
        """Extract faces using DeepFace models."""
        faces_data = []
        
        if not DEEPFACE_AVAILABLE or not self.available_deepface_models:
            return faces_data
        
        try:
            # Extract faces with alignment
            face_regions = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=False,
                align=True,
                normalize_face=True
            )
            
            for i, face_region in enumerate(face_regions):
                if face_region is not None:
                    # Calculate quality
                    face_img = (face_region * 255).astype(np.uint8)
                    quality_score = self.calculate_image_quality(face_img)
                    
                    if quality_score >= self.quality_threshold:
                        # Generate embeddings from multiple models
                        embeddings = {}
                        
                        for model_name in self.available_deepface_models:
                            try:
                                embedding = DeepFace.represent(
                                    img_path=face_img,
                                    model_name=model_name,
                                    enforce_detection=False,
                                    normalize=True
                                )
                                
                                if embedding and len(embedding) > 0:
                                    embeddings[model_name] = np.array(embedding[0]['embedding'])
                                    self.stats['deepface_embeddings'] += 1
                                    
                            except Exception as e:
                                logger.warning(f"Embedding failed for {model_name}: {e}")
                        
                        if embeddings:
                            # Create ensemble embedding (average of available models)
                            ensemble_embedding = np.mean(list(embeddings.values()), axis=0)
                            
                            face_data = {
                                'name': person_name,
                                'image_path': image_path,
                                'face_index': i,
                                'quality_score': quality_score,
                                'ensemble_embedding': ensemble_embedding,
                                'model_embeddings': embeddings,
                                'source': 'deepface',
                                'models_used': list(embeddings.keys())
                            }
                            
                            faces_data.append(face_data)
                            self.stats['multi_model_embeddings'] += 1
                            logger.info(f"✓ Multi-model embedding created for {person_name} (Q:{quality_score:.3f})")
                        
                        self.stats['quality_passed'] += 1
                    else:
                        logger.info(f"⚠ Face quality too low: {quality_score:.3f} < {self.quality_threshold}")
                
                self.stats['faces_detected'] += 1
        
        except Exception as e:
            logger.warning(f"DeepFace processing failed for {image_path}: {e}")
        
        return faces_data
    
    def extract_faces_insightface(self, image_path, person_name):
        """Extract faces using InsightFace as fallback."""
        faces_data = []
        
        if not self.insightface_app:
            return faces_data
        
        try:
            # Load and process image
            image = cv.imread(image_path)
            if image is None:
                return faces_data
            
            faces = self.insightface_app.get(image)
            
            for i, face in enumerate(faces):
                # Calculate quality
                bbox = face.bbox.astype(int)
                face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                quality_score = self.calculate_image_quality(face_region)
                
                if quality_score >= self.quality_threshold:
                    face_data = {
                        'name': person_name,
                        'image_path': image_path,
                        'face_index': i,
                        'quality_score': quality_score,
                        'ensemble_embedding': face.embedding,
                        'model_embeddings': {'insightface': face.embedding},
                        'source': 'insightface',
                        'models_used': ['insightface']
                    }
                    
                    faces_data.append(face_data)
                    self.stats['insightface_embeddings'] += 1
                    logger.info(f"✓ InsightFace embedding created for {person_name} (Q:{quality_score:.3f})")
                    self.stats['quality_passed'] += 1
                else:
                    logger.info(f"⚠ Face quality too low: {quality_score:.3f} < {self.quality_threshold}")
                
                self.stats['faces_detected'] += 1
        
        except Exception as e:
            logger.warning(f"InsightFace processing failed for {image_path}: {e}")
        
        return faces_data
    
    def process_person_folder(self, person_name, person_folder_path):
        """Process all images for a single person."""
        logger.info(f"📂 Processing person: {person_name}")
        
        person_faces = []
        image_files = [f for f in os.listdir(person_folder_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        for image_file in image_files:
            image_path = os.path.join(person_folder_path, image_file)
            logger.info(f"  📷 Processing: {image_file}")
            
            self.stats['total_images'] += 1
            
            # Try DeepFace first
            deepface_faces = self.extract_faces_deepface(image_path, person_name)
            
            if deepface_faces:
                person_faces.extend(deepface_faces)
                self.stats['processed_images'] += 1
            else:
                # Fall back to InsightFace
                logger.info(f"  🔄 Falling back to InsightFace for {image_file}")
                insightface_faces = self.extract_faces_insightface(image_path, person_name)
                if insightface_faces:
                    person_faces.extend(insightface_faces)
                    self.stats['processed_images'] += 1
                else:
                    logger.warning(f"  ❌ No faces extracted from {image_file}")
        
        logger.info(f"✅ Person '{person_name}' completed: {len(person_faces)} face embeddings")
        self.stats['people_processed'] += 1
        return person_faces
    
    def build_ultra_robust_database(self):
        """Build ultra-robust face database with multiple models."""
        logger.info("🚀 Starting ultra-robust face database creation...")
        
        if not os.path.exists(self.face_database_folder):
            logger.error(f"❌ Face database folder not found: {self.face_database_folder}")
            return False
        
        all_faces_data = []
        
        # Process each person's folder
        for person_folder in os.listdir(self.face_database_folder):
            person_folder_path = os.path.join(self.face_database_folder, person_folder)
            
            if os.path.isdir(person_folder_path):
                person_faces = self.process_person_folder(person_folder, person_folder_path)
                all_faces_data.extend(person_faces)
        
        if not all_faces_data:
            logger.error("❌ No face data extracted from any images!")
            return False
        
        # Prepare database structure
        database = {
            'version': '2.0_ultra_robust',
            'creation_date': str(np.datetime64('now')),
            'models_used': self.available_deepface_models + (['insightface'] if self.insightface_app else []),
            'face_data': all_faces_data,
            'names': [face['name'] for face in all_faces_data],
            'embeddings': [face['ensemble_embedding'] for face in all_faces_data],
            'quality_scores': [face['quality_score'] for face in all_faces_data],
            'sources': [face['source'] for face in all_faces_data],
            'statistics': self.stats.copy()
        }
        
        # Calculate quality statistics
        qualities = database['quality_scores']
        database['quality_stats'] = {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'median_quality': np.median(qualities)
        }
        
        # Save database
        try:
            with open(self.output_file, 'wb') as f:
                pickle.dump(database, f)
            
            logger.info(f"✅ Ultra-robust database saved: {self.output_file}")
            self.stats['embeddings_created'] = len(all_faces_data)
            
            # Print summary
            self.print_database_summary(database)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save database: {e}")
            return False
    
    def print_database_summary(self, database):
        """Print comprehensive database summary."""
        print("\n🎯 === Ultra-Robust Face Database Summary ===")
        print(f"📊 Total Face Embeddings: {len(database['embeddings'])}")
        print(f"👥 Unique People: {len(set(database['names']))}")
        print(f"🤖 Models Used: {', '.join(database['models_used'])}")
        print(f"📈 Average Quality Score: {database['quality_stats']['mean_quality']:.3f}")
        print(f"📉 Quality Range: {database['quality_stats']['min_quality']:.3f} - {database['quality_stats']['max_quality']:.3f}")
        
        print("\n📈 Processing Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Quality Distribution:")
        qualities = database['quality_scores']
        high_quality = sum(1 for q in qualities if q >= 0.7)
        medium_quality = sum(1 for q in qualities if 0.5 <= q < 0.7)
        low_quality = sum(1 for q in qualities if q < 0.5)
        
        print(f"  High Quality (≥0.7): {high_quality} ({high_quality/len(qualities)*100:.1f}%)")
        print(f"  Medium Quality (0.5-0.7): {medium_quality} ({medium_quality/len(qualities)*100:.1f}%)")
        print(f"  Low Quality (<0.5): {low_quality} ({low_quality/len(qualities)*100:.1f}%)")
        
        print("\n🎨 Source Distribution:")
        sources = database['sources']
        deepface_count = sum(1 for s in sources if s == 'deepface')
        insightface_count = sum(1 for s in sources if s == 'insightface')
        
        print(f"  DeepFace Embeddings: {deepface_count} ({deepface_count/len(sources)*100:.1f}%)")
        print(f"  InsightFace Embeddings: {insightface_count} ({insightface_count/len(sources)*100:.1f}%)")
        
        print("\n✅ Ultra-robust database creation completed!")

def main():
    """Main function for building ultra-robust face database."""
    print("🚀 === Ultra-Robust Face Database Builder ===")
    print("🎯 Features: Multi-Model Embeddings | Quality Assessment | Fallback Support")
    
    # Initialize builder
    builder = UltraRobustDatabaseBuilder(
        face_database_folder="face_database",
        output_file="ultra_robust_face_database.pkl",
        primary_models=['ArcFace', 'Facenet'],  # Best combination
        fallback_model='buffalo_l',
        min_face_size=80,
        quality_threshold=0.3
    )
    
    # Check if models are available
    if not DEEPFACE_AVAILABLE and not INSIGHTFACE_AVAILABLE:
        print("❌ Neither DeepFace nor InsightFace available. Please install dependencies:")
        print("   pip install deepface tensorflow insightface")
        return
    
    if not DEEPFACE_AVAILABLE:
        print("⚠ DeepFace not available - using InsightFace only")
        print("  Install DeepFace for ultra-robust features: pip install deepface tensorflow")
    
    # Build database
    success = builder.build_ultra_robust_database()
    
    if success:
        print("\n🎉 Ultra-robust face database created successfully!")
        print(f"📁 Database file: {builder.output_file}")
        print("🚀 Ready for ultra-robust face recognition!")
    else:
        print("\n❌ Database creation failed!")

if __name__ == "__main__":
    main()
