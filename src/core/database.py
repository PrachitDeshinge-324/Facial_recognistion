"""Database operations for face recognition system."""

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.config.paths import Database_Dir, Database_Path, Metadata_Path


class FaceDatabase:
    """Database operations for face recognition system."""
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    def __init__(self, database_path: Path = Database_Path, metadata_path: Path = Metadata_Path):
        """Initialize the database handler.
        
        Args:
            database_path: Path to the database file
            metadata_path: Path to the metadata file
        """
        self.database_path = database_path
        self.metadata_path = metadata_path
        self.database = None
        
    def load(self) -> bool:
        """Load the face database from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.database_path.exists():
            print(f"Database file not found: {self.database_path}")
            return False
            
        try:
            with open(self.database_path, 'rb') as f:
                self.database = pickle.load(f)
            
            print(f"Face database loaded successfully!")
            print(f"Database contains {len(self.database['names'])} embeddings")
            print(f"Unique people: {len(set(self.database['names']))}")
            
            # Print database summary
            for name in set(self.database['names']):
                count = self.database['names'].count(name)
                print(f"  {name}: {count} embeddings")
            
            return True
            
        except Exception as e:
            print(f"Error loading face database: {e}")
            return False
    
    def save(self, data: Dict) -> bool:
        """Save the database to disk.
        
        Args:
            data: Database dictionary to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Face database saved to {self.database_path}")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def create_metadata(self, names: Sequence[str], failed_images: int, source_directory: Path) -> bool:
        """Create metadata for the database.
        
        Args:
            names: List of identity names in the database
            failed_images: Count of images that failed processing
            source_directory: Source directory for the database
            
        Returns:
            True if metadata created successfully
        """
        try:
            summary = Counter(names)
            metadata = {
                "total_embeddings": len(names),
                "unique_identities": len(summary),
                "per_identity_counts": dict(summary),
                "failed_images": failed_images,
                "source_directory": str(source_directory.resolve()),
                "database_file": str(self.database_path.resolve()),
            }

            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Metadata written to {self.metadata_path}")
            return True
        except Exception as e:
            print(f"Error creating metadata: {e}")
            return False
    
    def get_database(self) -> Optional[Dict]:
        """Get the loaded database.
        
        Returns:
            Database dictionary or None if not loaded
        """
        return self.database