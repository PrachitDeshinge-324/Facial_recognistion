# Face Recognition System

A facial recognition system built with InsightFace and ONNX Runtime, featuring automatic hardware acceleration selection, database management, and real-time recognition capabilities.

## Features

- High-accuracy face detection and recognition
- Automatic hardware acceleration (CUDA, CoreML, CPU)
- Face database management tools
- Real-time video processing
- Support for headless operation

## Requirements

- Python 3.8+
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face_recognistionmodular.git
   cd Face_recognistionmodular
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Building Face Database

Before running recognition, build a face database from your images:

```bash
python script/build_database.py
```

Organize your face images in the following structure:
```
face_database/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
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

Key configuration files:
- `src/config/model.py`: Detection models and thresholds
- `src/config/paths.py`: Database and video paths

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