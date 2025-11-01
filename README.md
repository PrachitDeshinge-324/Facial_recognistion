# Face Recognition Pipeline

This repository contains a two-phase InsightFace-based workflow:

1. **Database build** – extract face embeddings from curated images and persist them for fast lookup.
2. **Real-time recognition** – stream video frames, detect faces, and recognise identities against the database.

## Installation
```bash
pip install -r requirements.txt --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```
## Quick start

```bash
python build_face_database.py --visualize  # optional visualisations
python main_recognition.py
python src/validate_database.py             # quick structural check
python src/main.py                          # detection-only demo
```

## Repository layout

```
.
├── build_face_database.py      # Phase 1: embedding generation
├── main_recognition.py         # Phase 2: real-time recognition loop
├── utils.py                    # InsightFace utilities and helpers
├── requirements.txt            # Runtime dependencies
├── README.md                   # Project overview (this file)
├── config/                     # Python configuration modules
│   ├── __init__.py             # Config package initialization
│   ├── models.py               # Model and processing configurations
│   └── paths.py                # Path configurations
├── src/                        # Shared libraries & auxiliary scripts
│   ├── main.py                 # Detector-only playground
│   └── validate_database.py    # Database structure sanity check
├── face_database/              # Source images and generated embeddings
├── face_detection/             # Auxiliary detection experiments
├── samples/                    # Example inputs for quick tests
├── output/                     # Generated artefacts (safe to delete)
├── docs/                       # Notes, figures, and PDF references
└── archives/                   # Legacy experiments & DeepFace workflows
```

Use the `output/` directory as scratch space—its contents are excluded from version control.

## Configuration

Configuration is now managed through Python modules in the `config/` directory:

- **`config/models.py`** - Model settings, thresholds, providers, and processing options
- **`config/paths.py`** - File paths, directories, and output locations

To modify settings, simply edit the relevant Python configuration file. All entry points automatically use these centralized configurations.

## Legacy DeepFace workflow

Historic DeepFace-based tooling has been preserved under `archives/legacy_deepface/`. The current InsightFace pipeline no longer depends on those scripts.
