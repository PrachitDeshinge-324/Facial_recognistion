Primary_model = "antelopev2"
Fallback_model = "buffalo_l"
Detection_size = 640

def get_available_providers():
    """Detect and return available ONNX Runtime providers based on system hardware"""
    import onnxruntime as ort
    
    # Get all available providers from onnxruntime
    available_providers = ort.get_available_providers()
    selected_providers = []
    
    # Prioritize providers in a sensible order
    if "CUDAExecutionProvider" in available_providers:
        selected_providers.append("CUDAExecutionProvider")
    
    if "CoreMLExecutionProvider" in available_providers:
        selected_providers.append("CoreMLExecutionProvider")
    
    # Always include CPU as fallback
    if "CPUExecutionProvider" in available_providers:
        selected_providers.append("CPUExecutionProvider")
    
    # If no providers were selected (unlikely), use default provider
    if not selected_providers:
        selected_providers = ["CPUExecutionProvider"]
    
    return selected_providers

# Set providers based on available hardware
Providers = get_available_providers()

Base_Threshold = 0.3