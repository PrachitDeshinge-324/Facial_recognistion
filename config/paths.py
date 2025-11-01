"""Path configurations for the face recognition system."""

from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database and storage paths
DATABASE_DIRECTORY = PROJECT_ROOT / "face_database"  # Source images folder
OUTPUT_DATABASE = PROJECT_ROOT / "database" / "face_database.pkl"  # Generated database file
METADATA_FILE = PROJECT_ROOT / "database" / "face_database_metadata.json"  # Database metadata

# Logging
LOG_FILE = PROJECT_ROOT / "face_recognition.log"

# Video and output paths
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "input" / "video.mp4"
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "output" / "output_video.mp4"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Source directory
SRC_DIR = PROJECT_ROOT / "src"