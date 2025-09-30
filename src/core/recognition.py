"""Face recognition module using embeddings comparison."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config.model import Base_Threshold


class FaceRecognizer:
    """Face recognition class that matches embeddings to known identities."""
    
    def __init__(self, 
                 database: Dict = None, 
                 threshold: float = Base_Threshold):
        """Initialize the face recognizer.
        
        Args:
            database: Dictionary containing face database
            threshold: Similarity threshold for recognition
        """
        self.database = database
        self.threshold = threshold
    
    def set_database(self, database: Dict) -> None:
        """Set or update the face database."""
        self.database = database
    
    def set_threshold(self, threshold: float) -> None:
        """Set the recognition threshold."""
        self.threshold = threshold
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize a face by comparing its embedding with the database.
        
        Args:
            face_embedding: The face embedding vector
            
        Returns:
            Tuple of (recognized_name, confidence_score)
        """
        if self.database is None or self.database.get('embeddings') is None or self.database.get('embeddings').size == 0:
            return "Unknown", 0.0

        # Calculate cosine similarity with all database embeddings
        similarities = cosine_similarity([face_embedding], self.database['embeddings'])[0]
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Check if the best match exceeds the threshold
        if best_similarity >= self.threshold:
            recognized_name = self.database['names'][best_match_idx]
            return recognized_name, best_similarity
        else:
            return "Unknown", best_similarity
    
    def recognize_faces(self, faces) -> List[Tuple[str, float]]:
        """Recognize multiple faces from InsightFace face objects.
        
        Args:
            faces: List of face objects from InsightFace
            
        Returns:
            List of (name, confidence) tuples for each face
        """
        results = []
        for face in faces:
            name, confidence = self.recognize_face(face.embedding)
            results.append((name, confidence))
        return results