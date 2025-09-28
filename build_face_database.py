"""
Phase 1: Face Database Builder.

This script processes images in the configured `face_database` folder, extracts
facial embeddings with InsightFace, and stores them for later recognition. The
refined implementation centralises configuration handling, records metadata, and
makes heavy visualisations optional.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2 as cv
import numpy as np

from src.config_utils import load_config
from src.insightface_utils import AnalysisConfig, create_face_analysis

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


@dataclass(frozen=True)
class DatabasePaths:
    root: Path
    output: Path
    metadata: Path


class FaceDatabaseBuilder:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        model_config = config.get("model_config", {})
        path_config = config.get("paths", {})

        defaults = AnalysisConfig()
        analysis_config = AnalysisConfig(
            model_name=model_config.get("primary_model", defaults.model_name),
            providers=tuple(model_config.get("providers", defaults.providers)),
            det_size=tuple(model_config.get("detection_size", defaults.det_size)),
            ctx_id=model_config.get("ctx_id", defaults.ctx_id),
        )

        print("Initializing InsightFace model...")
        self.app = create_face_analysis(analysis_config)
        print("Model initialised successfully!")

        root = Path(path_config.get("database_directory", "face_database"))
        output = Path(path_config.get("output_database", "database/face_database_antelopev2.pkl"))
        metadata = Path(path_config.get("metadata_file", "database/face_database_metadata.json"))
        self.paths = DatabasePaths(root=root, output=output, metadata=metadata)

    def extract_face_embedding(self, image_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract the first face embedding (and landmarks) from the given image."""

        try:
            img = cv.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None, None

            faces = self.app.get(img)
            if not faces:
                print(f"Warning: No face detected in {image_path}")
                return None, None
            if len(faces) > 1:
                print(f"Warning: Multiple faces detected in {image_path}, using the first one")

            face = faces[0]
            return face.embedding, face.landmark_3d_68
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error processing {image_path}: {exc}")
            return None, None

    def build_database(
        self,
        database_path: Path | None = None,
        *,
        visualize: bool = False,
    ) -> bool:
        """Process each person directory and persist the resulting database."""

        database_dir = Path(database_path) if database_path else self.paths.root
        if not database_dir.exists():
            print(f"Error: Database path {database_dir} does not exist")
            return False

        print(f"Building face database from {database_dir.resolve()}")

        names: list[str] = []
        embeddings: list[np.ndarray] = []
        landmarks: list[np.ndarray | None] = []
        image_paths: list[str] = []
        failed_images = 0

        for person_folder in sorted(database_dir.iterdir()):
            if not person_folder.is_dir():
                continue

            person_name = person_folder.name
            print(f"\nProcessing person: {person_name}")
            person_image_count = 0

            for image_file in sorted(person_folder.iterdir()):
                if image_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                print(f"  Processing: {image_file.name}")
                embedding, landmarks_3d = self.extract_face_embedding(image_file)
                if embedding is None:
                    print(f"    Failed to extract embedding from {image_file.name}")
                    failed_images += 1
                    continue

                names.append(person_name)
                embeddings.append(embedding.astype(np.float32))
                landmarks.append(landmarks_3d)
                image_paths.append(str(image_file))
                person_image_count += 1

            if person_image_count:
                print(f"  Successfully processed {person_image_count} images for {person_name}")
            else:
                print(f"  Warning: No valid embeddings found for {person_name}")

        if not embeddings:
            print("Error: No valid embeddings found in the database")
            return False

        try:
            embedding_array = np.stack(embeddings)
        except ValueError:
            embedding_array = np.array(embeddings, dtype=np.float32)

        face_database = {
            "names": names,
            "embeddings": embedding_array,
            "landmarks_3d": landmarks,
            "image_paths": image_paths,
        }

        self._save_database(face_database)
        self._write_metadata(names, failed_images, database_dir)

        if visualize:
            self._visualise_embeddings(face_database)

        return True

    def _save_database(self, face_database: Mapping[str, Any]) -> None:
        self.paths.output.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.output.open("wb") as handle:
            pickle.dump(face_database, handle)

        print(f"\nFace database saved to {self.paths.output}")
        print(f"Total entries: {len(face_database['names'])}")
        print(f"Unique people: {len(set(face_database['names']))}")

    def _write_metadata(self, names: Sequence[str], failed_images: int, database_dir: Path) -> None:
        summary = Counter(names)
        metadata = {
            "total_embeddings": len(names),
            "unique_identities": len(summary),
            "per_identity_counts": dict(summary),
            "failed_images": failed_images,
            "source_directory": str(database_dir.resolve()),
            "database_file": str(self.paths.output.resolve()),
        }

        self.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Metadata written to {self.paths.metadata}")

    def _visualise_embeddings(self, face_database: Mapping[str, Any]) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  pylint: disable=unused-import
            from sklearn.decomposition import PCA
        except ImportError as exc:
            print(f"Visualisation skipped: {exc}")
            return

        embeddings = face_database["embeddings"]
        names = face_database["names"]

        print("\nVisualising embeddings with PCA...")
        pca_2d = PCA(n_components=2)
        reduced_embeddings_2d = pca_2d.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_embeddings_2d[:, 0],
            y=reduced_embeddings_2d[:, 1],
            hue=names,
            palette="tab10",
            s=100,
        )
        plt.title("2D PCA of Face Embeddings")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        pca_3d = PCA(n_components=3)
        reduced_embeddings_3d = pca_3d.fit_transform(embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            reduced_embeddings_3d[:, 0],
            reduced_embeddings_3d[:, 1],
            reduced_embeddings_3d[:, 2],
            c=[hash(name) % 10 for name in names],
            cmap="tab10",
            s=100,
        )
        ax.set_title("3D PCA of Face Embeddings")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        plt.tight_layout()
        plt.show()

        print("\nVisualising 3D face landmarks for first face per person...")
        connections = [
            list(range(0, 17)),
            list(range(17, 22)),
            list(range(22, 27)),
            list(range(27, 31)),
            list(range(31, 36)),
            list(range(36, 42)),
            [36, 41],
            list(range(42, 48)),
            [42, 47],
            list(range(48, 60)),
            [48, 59],
            list(range(60, 68)),
            [60, 67],
        ]

        seen: set[str] = set()
        for name, landmarks, image_path in zip(
            names, face_database["landmarks_3d"], face_database.get("image_paths", [])
        ):
            if name in seen or landmarks is None:
                continue
            seen.add(name)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(landmarks[:, 0], landmarks[:, 2], -landmarks[:, 1], c="b", s=15)
            for conn in connections:
                ax.plot(landmarks[conn, 0], landmarks[conn, 2], -landmarks[conn, 1], c="r", linewidth=2)
            ax.set_title(f"3D Landmarks for {name} (First Face)")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("-Y")
            ax.view_init(elev=0, azim=-90)
            plt.tight_layout()
            plt.show()

            if image_path:
                try:
                    img = cv.imread(image_path)
                    if img is None:
                        continue
                    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    landmarks_2d = landmarks[:, :2].astype(int)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img_rgb)
                    plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c="b", s=15)
                    for conn in connections:
                        plt.plot(landmarks_2d[conn, 0], landmarks_2d[conn, 1], c="r", linewidth=2)
                    plt.title(f"2D Landmarks Overlay for {name}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Could not plot 2D overlay for {name}: {exc}")

    def summary(self) -> dict[str, Any]:
        return {
            "database": str(self.paths.output.resolve()),
            "metadata": str(self.paths.metadata.resolve()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the InsightFace database")
    parser.add_argument("--database-path", type=Path, default=None, help="Override the database input directory")
    parser.add_argument("--config", type=Path, default=Path("config/config.json"), help="Path to the configuration file")
    parser.add_argument("--visualize", action="store_true", help="Enable PCA and landmark visualisations")
    return parser.parse_args()


def main() -> None:
    print("=== Face Database Builder ===")
    print("Ensure your folder structure matches the expected layout:")
    print("face_database/ -> Person/ -> image1.jpg")

    args = parse_args()
    config = load_config(args.config)

    builder = FaceDatabaseBuilder(config=config)
    database_dir = args.database_path or builder.paths.root

    if not database_dir.exists():
        print(f"Error: face database directory '{database_dir}' not found!")
        print("Please create the directory and add person folders with images.")
        return

    success = builder.build_database(database_dir, visualize=args.visualize)

    if success:
        print("\n✅ Face database built successfully!")
        print(f"Embeddings stored in: {builder.paths.output}")
        print(f"Metadata stored in: {builder.paths.metadata}")
        print("You can now run the recognition system with the updated pipeline.")
    else:
        print("\n❌ Failed to build face database")
        print("Please ensure that you have valid images in the person folders.")


if __name__ == "__main__":
    main()
