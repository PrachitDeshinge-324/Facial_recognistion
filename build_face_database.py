"""
Phase 1: Face Database Builder
This script processes images in the face_database folder, extracts facial embeddings,
and saves them for later recognition.
"""

import os
import cv2 as cv
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path

class FaceDatabaseBuilder:
    def __init__(self):
        """Initialize the face analysis model."""
        print("Initializing FaceAnalysis model...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model initialized successfully!")
        
    def extract_face_embedding(self, image_path):
        """
        Extract face embedding from a single image.
        Returns the embedding and 3D landmarks if a face is found, None otherwise.
        """
        try:
            # Read the image
            img = cv.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None, None
            
            # Get faces from the image
            faces = self.app.get(img)
            
            if len(faces) == 0:
                print(f"Warning: No face detected in {image_path}")
                return None, None
            elif len(faces) > 1:
                print(f"Warning: Multiple faces detected in {image_path}, using the first one")
            
            # Return the embedding and 3D landmarks of the first (or only) face
            face = faces[0]
            embedding = face.embedding
            landmarks_3d = face.landmark_3d_68  # 68-point 3D landmarks (numpy array)
            return embedding, landmarks_3d
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, None
    
    def build_database(self, database_path="face_database"):
        """
        Build the face database by processing all images in person folders.
        """
        database_path = Path(database_path)
        if not database_path.exists():
            print(f"Error: Database path {database_path} does not exist")
            return False
        
        face_database = {
            'names': [],
            'embeddings': [],
            'landmarks_3d': []  # New: store 3D landmarks
        }
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        print(f"Building face database from {database_path}")
        
        # Loop through each person's folder
        for person_folder in database_path.iterdir():
            if not person_folder.is_dir():
                continue
                
            person_name = person_folder.name
            print(f"\nProcessing person: {person_name}")
            
            person_embeddings = []
            person_landmarks = []  # New: store landmarks for this person
            image_count = 0
            
            # Process all images in this person's folder
            for image_file in person_folder.iterdir():
                if image_file.suffix.lower() in image_extensions:
                    print(f"  Processing: {image_file.name}")
                    
                    embedding, landmarks_3d = self.extract_face_embedding(str(image_file))
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        person_landmarks.append(landmarks_3d)  # Store landmarks
                        image_count += 1
                    else:
                        print(f"    Failed to extract embedding from {image_file.name}")
            
            if person_embeddings:
                # Store all embeddings and landmarks for this person
                for embedding, landmarks in zip(person_embeddings, person_landmarks):
                    face_database['names'].append(person_name)
                    face_database['embeddings'].append(embedding)
                    face_database['landmarks_3d'].append(landmarks)
                
                print(f"  Successfully processed {image_count} images for {person_name}")
            else:
                print(f"  Warning: No valid embeddings found for {person_name}")
        
        # Convert embeddings to numpy array for efficient computation
        if face_database['embeddings']:
            face_database['embeddings'] = np.array(face_database['embeddings'])
            
            # Save the database
            database_file = "database/face_database_buffalo.pkl"
            with open(database_file, 'wb') as f:
                pickle.dump(face_database, f)
            
            print(f"\nFace database saved to {database_file}")
            print(f"Total entries: {len(face_database['names'])}")
            print(f"Unique people: {len(set(face_database['names']))}")
            
            # Print summary
            for name in set(face_database['names']):
                count = face_database['names'].count(name)
                print(f"  {name}: {count} embeddings")
            
            # Visualizations (2D/3D embeddings and 3D faces)
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
            
            print("\nVisualizing embeddings with PCA...")
            
            # 2D PCA (existing, kept as-is)
            pca_2d = PCA(n_components=2)
            reduced_embeddings_2d = pca_2d.fit_transform(face_database['embeddings'])
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=reduced_embeddings_2d[:, 0], y=reduced_embeddings_2d[:, 1], hue=face_database['names'], palette='tab10', s=100)
            plt.title('2D PCA of Face Embeddings')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # 3D PCA (new addition)
            pca_3d = PCA(n_components=3)
            reduced_embeddings_3d = pca_3d.fit_transform(face_database['embeddings'])
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embeddings_3d[:, 0], reduced_embeddings_3d[:, 1], reduced_embeddings_3d[:, 2], 
                                 c=[hash(name) % 10 for name in face_database['names']], cmap='tab10', s=100)
            ax.set_title('3D PCA of Face Embeddings')
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_zlabel('PCA Component 3')
            # Add legend (simplified)
            unique_names = list(set(face_database['names']))
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), markersize=10) for i in range(len(unique_names))]
            ax.legend(handles, unique_names, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # 3D Face Landmark Visualization (improved: connect points to show face structure)
            print("\nVisualizing 3D face landmarks for first face per person...")
            # Standard 68-point landmark connections
            connections = [
                list(range(0, 17)),      # Jaw line
                list(range(17, 22)),     # Right eyebrow
                list(range(22, 27)),     # Left eyebrow
                list(range(27, 31)),     # Nose bridge
                list(range(31, 36)),     # Lower nose
                list(range(36, 42)),     # Right eye
                [36, 41],                # Right eye close
                list(range(42, 48)),     # Left eye
                [42, 47],                # Left eye close
                list(range(48, 60)),     # Outer lip
                [48, 59],                # Outer lip close
                list(range(60, 68)),     # Inner lip
                [60, 67],                # Inner lip close
            ]
            shown_names = set()
            for name, landmarks in zip(face_database['names'], face_database['landmarks_3d']):
                if name in shown_names:
                    continue  # Only show one face per person
                shown_names.add(name)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                # Swap axes for better frontal view
                ax.scatter(landmarks[:, 0], landmarks[:, 2], -landmarks[:, 1], c='b', s=15, label='Landmark Points')
                for conn in connections:
                    ax.plot(landmarks[conn, 0], landmarks[conn, 2], -landmarks[conn, 1], c='r', linewidth=2)
                ax.set_title(f'3D Landmarks for {name} (First Face)')
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_zlabel('-Y')
                ax.view_init(elev=0, azim=-90)  # Frontal view
                plt.legend()
                plt.tight_layout()
                plt.show()
                
                # Overlay 2D landmarks on the original image (optional)
                try:
                    img_path = None
                    # Find the first image for this person
                    for person_folder in database_path.iterdir():
                        if person_folder.name == name:
                            for image_file in person_folder.iterdir():
                                if image_file.suffix.lower() in image_extensions:
                                    img_path = str(image_file)
                                    break
                            break
                    if img_path:
                        img = cv.imread(img_path)
                        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                        # Project 3D landmarks to 2D (ignore Z)
                        landmarks_2d = landmarks[:, :2].astype(int)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(img_rgb)
                        plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='b', s=15)
                        for conn in connections:
                            plt.plot(landmarks_2d[conn, 0], landmarks_2d[conn, 1], c='r', linewidth=2)
                        plt.title(f'2D Landmarks Overlay for {name}')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                except Exception as e:
                    print(f"Could not plot 2D overlay for {name}: {e}")
            
            return True
        else:
            print("Error: No valid embeddings found in the database")
            return False

def main():
    """Main function to build the face database."""
    print("=== Face Database Builder ===")
    print("Make sure you have placed images in the face_database folder structure:")
    print("face_database/")
    print("├── Person1_Name/")
    print("│   ├── image1.jpg")
    print("│   └── image2.jpg")
    print("└── Person2_Name/")
    print("    ├── image1.jpg")
    print("    └── image2.jpg")
    print()
    
    # Check if face_database directory exists
    if not os.path.exists("face_database"):
        print("Error: face_database directory not found!")
        print("Please create the directory and add person folders with images.")
        return
    
    # Build the database
    builder = FaceDatabaseBuilder()
    success = builder.build_database()
    
    if success:
        print("\n✅ Face database built successfully!")
        print("You can now run the recognition system with the updated main.py")
    else:
        print("\n❌ Failed to build face database")
        print("Please check that you have valid images in the person folders")

if __name__ == "__main__":
    main()
