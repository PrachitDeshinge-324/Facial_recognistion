# Face Recognition Pipeline

This repository contains a two-phase InsightFace-based workflow:

1. **Database build** – extract face embeddings from curated images and persist them for fast lookup.
2. **Real-time recognition** – stream video frames, detect faces, and recognise identities against the database.

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
├── requirements.txt            # Runtime dependencies
├── README.md                   # Project overview (this file)
├── config/                     # Shared runtime configuration
│   └── config.json
├── src/                        # Shared libraries & auxiliary scripts
│   ├── main.py                 # Detector-only playground
│   ├── validate_database.py    # Database structure sanity check
│   ├── config_utils.py         # Config loader used across entry-points
│   └── insightface_utils.py    # InsightFace helper utilities
├── database/                   # Generated embedding stores & metadata
├── face_database/              # Source images grouped by identity
├── face_detection/             # Auxiliary detection experiments
├── samples/                    # Example inputs for quick tests
├── output/                     # Generated artefacts (safe to delete)
├── docs/                       # Notes, figures, and PDF references
└── archives/                   # Legacy experiments & DeepFace workflows
```

Use the `output/` directory as scratch space—its contents are excluded from version control.

## Configuration

Runtime options live in `config/config.json`. All entry points share `src.config_utils.load_config`, so tweaks (such as model providers or output locations) propagate automatically.

## Legacy DeepFace workflow

Historic DeepFace-based tooling has been preserved under `archives/legacy_deepface/`. The current InsightFace pipeline no longer depends on those scripts.
