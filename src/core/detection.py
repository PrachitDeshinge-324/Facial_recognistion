"""Face detection module using InsightFace."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis

from src.config.model import Primary_model, Fallback_model, Detection_size, Providers


class FaceDetector:
    """Face detection class using InsightFace models."""
    
    def __init__(self, model_name: str = Primary_model, providers: List[str] = None):
        """Initialize the face detector.
        
        Args:
            model_name: Name of the InsightFace model to use
            providers: List of ONNX providers to try
        """
        self.model_name = model_name
        self.providers = providers or Providers
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the InsightFace model."""
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=(Detection_size, Detection_size))
        except Exception as e:
            print(f"Error initializing primary model: {e}")
            if self.model_name != Fallback_model:
                print(f"Trying fallback model: {Fallback_model}")
                self.model_name = Fallback_model
                self.app = FaceAnalysis(name=self.model_name, providers=self.providers)
                self.app.prepare(ctx_id=0, det_size=(Detection_size, Detection_size))
    
    def detect_faces(self, image: np.ndarray):
        """Detect faces in an image.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of face objects from InsightFace
        """
        if self.app is None:
            raise RuntimeError("Model not initialized")
        
        return self.app.get(image)

    def get_face_embeddings(self, image: np.ndarray):
        """Get face embeddings from an image.
        
        Returns the faces with their embeddings.
        """
        faces = self.detect_faces(image)
        return faces  # InsightFace already includes embeddings in face objects
    
    def extract_single_face(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract the first face embedding and landmarks from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (embedding, landmarks) or (None, None) if no face found
        """
        try:
            img = cv.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None, None

            faces = self.detect_faces(img)
            if not faces:
                print(f"Warning: No face detected in {image_path}")
                return None, None
            if len(faces) > 1:
                print(f"Warning: Multiple faces detected in {image_path}, using the first one")

            face = faces[0]
            return face.embedding, face.landmark_3d_68
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None