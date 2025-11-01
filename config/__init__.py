"""Configuration package for face recognition system."""

from .models import (
    AnalysisConfig,
    get_model_config,
    get_recognition_config,
    get_quality_config,
    get_processing_config,
)
from .paths import (
    PROJECT_ROOT,
    DATABASE_DIRECTORY,
    OUTPUT_DATABASE,
    METADATA_FILE,
    LOG_FILE,
    DEFAULT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    OUTPUT_DIRECTORY,
    CONFIG_DIR,
    SRC_DIR,
)

__all__ = [
    "AnalysisConfig",
    "get_model_config",
    "get_recognition_config",
    "get_quality_config", 
    "get_processing_config",
    "PROJECT_ROOT",
    "DATABASE_DIRECTORY",
    "OUTPUT_DATABASE",
    "METADATA_FILE",
    "LOG_FILE",
    "DEFAULT_VIDEO_PATH",
    "OUTPUT_VIDEO_PATH",
    "OUTPUT_DIRECTORY",
    "CONFIG_DIR",
    "SRC_DIR",
]