"""Utility to sanity-check the persisted face database."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = {"names", "embeddings"}
OPTIONAL_KEYS = {"landmarks_3d", "image_paths"}


def validate_database(database_path: Path) -> dict[str, Any]:
    if not database_path.exists():
        raise FileNotFoundError(f"Database file not found: {database_path}")

    with database_path.open("rb") as handle:
        data = pickle.load(handle)

    missing_keys = REQUIRED_KEYS - data.keys()
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")

    embeddings = np.asarray(data["embeddings"])
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    names = data["names"]
    if len(names) != len(embeddings):
        raise ValueError("Number of names does not match number of embeddings")

    optional_missing = OPTIONAL_KEYS - data.keys()
    summary = {
        "path": str(database_path.resolve()),
        "entries": len(names),
        "embedding_dim": embeddings.shape[1],
        "unique_people": len(set(names)),
        "missing_optional_keys": sorted(optional_missing),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a face database pickle file")
    parser.add_argument(
        "database",
        type=Path,
        nargs="?",
        default=Path("database/face_database_antelopev2.pkl"),
        help="Path to the database pickle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_database(args.database)
    print("Database summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
