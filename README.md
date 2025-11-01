# Face Recognition System

A facial recognition system built with InsightFace and ONNX Runtime, featuring automatic hardware acceleration selection, database management, and real-time recognition capabilities.

## Features

## Installation
```bash
pip install -r requirements.txt --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```
## Quick start

```bash
python script/build_database.py
```

Organize your face images in the following structure:
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

### Running Face Recognition

Process a video file or camera stream:

```bash
# Process default video file
python script/run_recognition.py

# Process specific video file
python script/run_recognition.py --video path/to/video.mp4

# Use camera (webcam)
python script/run_recognition.py --camera 0

# Specify output file
python script/run_recognition.py --output path/to/output.mp4

# Run without display (headless mode)
python script/run_recognition.py --no-display
```

## Configuration

Configuration is now managed through Python modules in the `config/` directory:

- **`config/models.py`** - Model settings, thresholds, providers, and processing options
- **`config/paths.py`** - File paths, directories, and output locations

To modify settings, simply edit the relevant Python configuration file. All entry points automatically use these centralized configurations.

## Project Structure

```
face_recognition_modular/
├── src/                      # Source code
│   ├── core/                 # Core functionality
│   ├── utils/                # Utility functions
│   └── config/               # Configuration
├── script/                   # Entry point scripts
├── face_database/            # Face image database
└── database/                 # Generated embeddings database
```



Created by Prachit Deshinge