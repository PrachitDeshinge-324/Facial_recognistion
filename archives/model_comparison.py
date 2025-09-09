"""
Face Recognition Model Comparison Tool
This script compares the performance of standard vs enhanced ArcFace models.
"""

import cv2 as cv
import numpy as np
import time
import pickle
import os
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self):
        """Initialize comparison tool with multiple models."""
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load different face recognition models for comparison."""
        logger.info("Loading models for comparison...")
        
        # Standard model (buffalo_l)
        try:
            self.models['buffalo_l'] = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.models['buffalo_l'].prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✓ Loaded buffalo_l model")
        except Exception as e:
            logger.warning(f"Failed to load buffalo_l: {e}")
        
        # Enhanced model (antelopev2)
        try:
            self.models['antelopev2'] = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
            self.models['antelopev2'].prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✓ Loaded antelopev2 model")
        except Exception as e:
            logger.warning(f"Failed to load antelopev2: {e}")
    
    def arcface_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate ArcFace-style similarity with angular margin."""
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        cosine_sim = np.dot(embedding1_norm, embedding2_norm)
        angular_margin = 0.5
        enhanced_similarity = cosine_sim - angular_margin
        
        return max(min(enhanced_similarity, 1.0), -1.0)
    
    def compare_models_on_images(self, image_paths: List[str]) -> Dict:
        """Compare model performance on a set of images."""
        results = {
            'buffalo_l': {'embeddings': [], 'processing_times': [], 'detection_scores': []},
            'antelopev2': {'embeddings': [], 'processing_times': [], 'detection_scores': []}
        }
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            img = cv.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
            
            logger.info(f"Processing: {os.path.basename(image_path)}")
            
            for model_name, model in self.models.items():
                start_time = time.time()
                
                try:
                    faces = model.get(img)
                    processing_time = time.time() - start_time
                    
                    if len(faces) > 0:
                        # Use the first detected face
                        face = faces[0]
                        embedding = face.embedding
                        det_score = face.det_score if hasattr(face, 'det_score') else 1.0
                        
                        results[model_name]['embeddings'].append(embedding)
                        results[model_name]['processing_times'].append(processing_time)
                        results[model_name]['detection_scores'].append(det_score)
                        
                        logger.info(f"  {model_name}: {processing_time:.3f}s, det_score: {det_score:.3f}")
                    else:
                        logger.warning(f"  {model_name}: No face detected")
                        
                except Exception as e:
                    logger.error(f"  {model_name}: Error - {e}")
        
        return results
    
    def calculate_similarity_matrices(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cosine and ArcFace similarity matrices."""
        n = len(embeddings)
        cosine_matrix = np.zeros((n, n))
        arcface_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cosine_matrix[i, j] = 1.0
                    arcface_matrix[i, j] = 1.0
                else:
                    # Cosine similarity
                    cosine_sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    cosine_matrix[i, j] = cosine_sim
                    
                    # ArcFace similarity
                    arcface_sim = self.arcface_similarity(embeddings[i], embeddings[j])
                    arcface_matrix[i, j] = arcface_sim
        
        return cosine_matrix, arcface_matrix
    
    def analyze_robustness(self, image_dir: str) -> Dict:
        """Analyze model robustness across different conditions."""
        logger.info(f"Analyzing robustness using images from: {image_dir}")
        
        # Find all images in directory
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend([str(p) for p in Path(image_dir).rglob(f'*{ext}')])
            image_paths.extend([str(p) for p in Path(image_dir).rglob(f'*{ext.upper()}')])
        
        if not image_paths:
            logger.error(f"No images found in {image_dir}")
            return {}
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Compare models
        results = self.compare_models_on_images(image_paths[:10])  # Limit for demo
        
        # Calculate statistics
        analysis = {}
        for model_name, data in results.items():
            if data['embeddings']:
                analysis[model_name] = {
                    'avg_processing_time': np.mean(data['processing_times']),
                    'std_processing_time': np.std(data['processing_times']),
                    'avg_detection_score': np.mean(data['detection_scores']),
                    'std_detection_score': np.std(data['detection_scores']),
                    'total_faces_detected': len(data['embeddings']),
                    'embedding_dimension': len(data['embeddings'][0]) if data['embeddings'] else 0
                }
                
                # Calculate similarity statistics if multiple embeddings
                if len(data['embeddings']) > 1:
                    cosine_matrix, arcface_matrix = self.calculate_similarity_matrices(data['embeddings'])
                    
                    # Get upper triangle (excluding diagonal)
                    upper_tri_indices = np.triu_indices_from(cosine_matrix, k=1)
                    cosine_similarities = cosine_matrix[upper_tri_indices]
                    arcface_similarities = arcface_matrix[upper_tri_indices]
                    
                    analysis[model_name].update({
                        'avg_cosine_similarity': np.mean(cosine_similarities),
                        'std_cosine_similarity': np.std(cosine_similarities),
                        'avg_arcface_similarity': np.mean(arcface_similarities),
                        'std_arcface_similarity': np.std(arcface_similarities)
                    })
        
        return analysis
    
    def generate_comparison_report(self, analysis: Dict) -> str:
        """Generate a detailed comparison report."""
        report = "\n" + "="*80 + "\n"
        report += "FACE RECOGNITION MODEL COMPARISON REPORT\n"
        report += "="*80 + "\n\n"
        
        for model_name, stats in analysis.items():
            report += f"Model: {model_name.upper()}\n"
            report += "-" * 50 + "\n"
            report += f"Faces Detected: {stats.get('total_faces_detected', 0)}\n"
            report += f"Embedding Dimension: {stats.get('embedding_dimension', 0)}\n"
            report += f"Avg Processing Time: {stats.get('avg_processing_time', 0):.4f}s ± {stats.get('std_processing_time', 0):.4f}s\n"
            report += f"Avg Detection Score: {stats.get('avg_detection_score', 0):.3f} ± {stats.get('std_detection_score', 0):.3f}\n"
            
            if 'avg_cosine_similarity' in stats:
                report += f"Avg Cosine Similarity: {stats['avg_cosine_similarity']:.3f} ± {stats['std_cosine_similarity']:.3f}\n"
                report += f"Avg ArcFace Similarity: {stats['avg_arcface_similarity']:.3f} ± {stats['std_arcface_similarity']:.3f}\n"
            
            report += "\n"
        
        # Comparison summary
        if len(analysis) > 1:
            report += "COMPARISON SUMMARY\n"
            report += "-" * 50 + "\n"
            
            models = list(analysis.keys())
            if len(models) == 2:
                model1, model2 = models
                
                time_diff = analysis[model1]['avg_processing_time'] - analysis[model2]['avg_processing_time']
                det_diff = analysis[model1]['avg_detection_score'] - analysis[model2]['avg_detection_score']
                
                report += f"Processing Time Difference: {time_diff:.4f}s\n"
                report += f"Detection Score Difference: {det_diff:.3f}\n"
                
                if 'avg_cosine_similarity' in analysis[model1] and 'avg_cosine_similarity' in analysis[model2]:
                    cosine_diff = analysis[model1]['avg_cosine_similarity'] - analysis[model2]['avg_cosine_similarity']
                    arcface_diff = analysis[model1]['avg_arcface_similarity'] - analysis[model2]['avg_arcface_similarity']
                    report += f"Cosine Similarity Difference: {cosine_diff:.3f}\n"
                    report += f"ArcFace Similarity Difference: {arcface_diff:.3f}\n"
        
        report += "\nRECOMMendations:\n"
        report += "-" * 50 + "\n"
        report += "• ArcFace models (antelopev2) typically provide better accuracy\n"
        report += "• Use adaptive thresholds based on face quality\n"
        report += "• Implement ensemble methods for critical applications\n"
        report += "• Regular model updates improve performance\n"
        
        return report

def main():
    """Main function for model comparison."""
    print("=== Face Recognition Model Comparison ===")
    
    # Initialize comparison tool
    comparator = ModelComparison()
    
    if not comparator.models:
        print("No models loaded successfully. Please check your InsightFace installation.")
        return
    
    # Run analysis on face database
    database_dir = "face_database"
    if os.path.exists(database_dir):
        analysis = comparator.analyze_robustness(database_dir)
        
        if analysis:
            # Generate and display report
            report = comparator.generate_comparison_report(analysis)
            print(report)
            
            # Save report to file
            with open("model_comparison_report.txt", "w") as f:
                f.write(report)
            print("Detailed report saved to: model_comparison_report.txt")
        else:
            print("Analysis failed. Please check your face database directory.")
    else:
        print(f"Face database directory '{database_dir}' not found.")
        print("Please ensure you have a face database set up.")

if __name__ == "__main__":
    main()
