"""Model and processing configurations for the face recognition system."""

from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any
import onnxruntime as ort

# Model Configuration
PRIMARY_MODEL = "antelopev2"
FALLBACK_MODEL = "buffalo_l"
DETECTION_SIZE = (640, 640)

# Provider priority order - InsightFace will automatically choose the best available
# Listed in order of preference (fastest to slowest)
PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",    # NVIDIA TensorRT (fastest GPU)
    "CUDAExecutionProvider",        # NVIDIA CUDA (fast GPU)
    "ROCMExecutionProvider",        # AMD ROCm (AMD GPU)
    "CoreMLExecutionProvider",      # Apple CoreML (Apple Silicon)
    "OpenVINOExecutionProvider",    # Intel OpenVINO (Intel hardware optimization)
    "DirectMLExecutionProvider",    # DirectML (Windows GPU acceleration)
    "CPUExecutionProvider"          # CPU fallback (always available)
]

def get_available_providers() -> list[str]:
    """Get the best available providers in priority order."""
    available_providers = set(ort.get_available_providers())
    
    # Filter and return providers in priority order
    selected_providers = [p for p in PROVIDER_PRIORITY if p in available_providers]
    
    print(f"🔍 Available ONNX providers: {list(available_providers)}")
    print(f"🚀 Selected providers (in priority order): {selected_providers}")
    
    # Determine the primary provider for optimization
    if selected_providers:
        primary_provider = selected_providers[0]
        if "CUDA" in primary_provider or "Tensorrt" in primary_provider:
            print("🎮 Using NVIDIA GPU acceleration")
        elif "CoreML" in primary_provider:
            print("🍎 Using Apple CoreML acceleration")
        elif "ROCM" in primary_provider:
            print("🔴 Using AMD GPU acceleration")
        elif "OpenVINO" in primary_provider:
            print("🔵 Using Intel OpenVINO acceleration")
        elif "DirectML" in primary_provider:
            print("🖥️ Using DirectML acceleration")
        else:
            print("💻 Using CPU execution")
    
    return selected_providers


def get_model_specific_providers(model_name: str) -> list[str]:
    """Get providers optimized for specific models."""
    available_providers = set(ort.get_available_providers())
    
    # Model-specific provider preferences
    model_preferences = {
        "antelopev2": [
            "CUDAExecutionProvider",        # Works best with CUDA
            "CPUExecutionProvider",         # CPU fallback for antelopev2
            # Note: CoreML often has issues with antelopev2
        ],
        "buffalo_l": [
            "CoreMLExecutionProvider",      # Works well with CoreML
            "CUDAExecutionProvider", 
            "CPUExecutionProvider"
        ],
        "buffalo_m": [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ],
        "buffalo_s": [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider", 
            "CPUExecutionProvider"
        ]
    }
    
    # Get model-specific preferences or fall back to default
    preferred_providers = model_preferences.get(model_name, PROVIDER_PRIORITY)
    
    # Filter by availability
    selected_providers = [p for p in preferred_providers if p in available_providers]
    
    print(f"🎯 Model-specific providers for {model_name}: {selected_providers}")
    return selected_providers


# Get the best available providers at module load time
DEFAULT_PROVIDERS = get_available_providers()

# Recognition Configuration
BASE_THRESHOLD = 0.3
CONFIDENCE_MODE = "adaptive"

ADAPTIVE_THRESHOLDS = {
    "high_quality": 0.5,
    "medium_quality": 0.4,
    "low_quality": 0.3
}

ENSEMBLE_WEIGHTS = {
    "cosine_similarity": 0.6,
    "arcface_similarity": 0.4
}

# Quality Configuration
MIN_FACE_QUALITY = 0.3

QUALITY_FACTORS = {
    "detection_confidence": 0.25,
    "face_size": 0.25,
    "center_position": 0.15,
    "pose_angle": 0.20,
    "age_factor": 0.15
}

# Processing Configuration
MAX_FACES_PER_FRAME = 10
FRAME_SKIP = 1
LOGGING_LEVEL = "INFO"
SHOW_DEBUG_INFO = True


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration needed to prepare an InsightFace analysis instance."""
    
    model_name: str = PRIMARY_MODEL
    providers: Sequence[str] = None
    det_size: Tuple[int, int] = DETECTION_SIZE
    ctx_id: int = 0
    
    def __post_init__(self):
        if self.providers is None:
            # Use the automatically detected best providers
            object.__setattr__(self, 'providers', DEFAULT_PROVIDERS)


def get_model_config() -> Dict[str, Any]:
    """Get model configuration as dictionary."""
    return {
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "detection_size": DETECTION_SIZE,
        "providers": DEFAULT_PROVIDERS
    }


def get_recognition_config() -> Dict[str, Any]:
    """Get recognition configuration as dictionary."""
    return {
        "base_threshold": BASE_THRESHOLD,
        "confidence_mode": CONFIDENCE_MODE,
        "adaptive_thresholds": ADAPTIVE_THRESHOLDS,
        "ensemble_weights": ENSEMBLE_WEIGHTS
    }


def get_quality_config() -> Dict[str, Any]:
    """Get quality configuration as dictionary."""
    return {
        "min_face_quality": MIN_FACE_QUALITY,
        "quality_factors": QUALITY_FACTORS
    }


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration as dictionary."""
    return {
        "max_faces_per_frame": MAX_FACES_PER_FRAME,
        "frame_skip": FRAME_SKIP,
        "logging_level": LOGGING_LEVEL,
        "show_debug_info": SHOW_DEBUG_INFO
    }


def get_available_models() -> Dict[str, str]:
    """Get information about available InsightFace models."""
    return {
        "buffalo_l": "Stable, widely compatible model - good for most use cases",
        "buffalo_m": "Medium model - balance between speed and accuracy", 
        "buffalo_s": "Small model - fastest but lower accuracy",
        "antelopev2": "Advanced model - may require specific providers/hardware",
        "webface600k_r50": "High accuracy model - larger and slower"
    }