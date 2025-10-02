import torch
import sys
import warnings


def check_gpu_available():
    """
    Check if GPU is available for PyTorch operations.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()


def get_device():
    """
    Get the best available device for PyTorch operations.

    Returns:
        torch.device: The device to use (cuda if available, otherwise cpu)
    """
    if check_gpu_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
        return device


def setup_model_for_gpu(model, device=None):
    """
    Setup a model for GPU or CPU inference.

    Args:
        model: PyTorch model to setup
        device (torch.device, optional): Device to use. If None, will auto-detect.

    Returns:
        tuple: (model, device) - The model moved to the appropriate device and the device used
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    if device.type == 'cuda':
        # Enable mixed precision for faster inference
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            model.use_amp = True
        else:
            model.use_amp = False
            warnings.warn("Mixed precision not available, using standard precision")

    return model, device


def get_optimal_batch_size(device_type='cpu', model_size='small'):
    """
    Get optimal batch size based on device and model size.

    Args:
        device_type (str): 'cuda' or 'cpu'
        model_size (str): 'small', 'medium', or 'large'

    Returns:
        int: Recommended batch size
    """
    if device_type == 'cuda':
        if model_size == 'small':
            return 32
        elif model_size == 'medium':
            return 16
        else:  # large
            return 8
    else:  # CPU
        if model_size == 'small':
            return 16
        elif model_size == 'medium':
            return 8
        else:  # large
            return 4


def optimize_inference_settings():
    """
    Optimize PyTorch settings for inference performance.
    """
    # Enable cuDNN auto-tuner for best performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Disable gradient calculation for inference
    torch.set_grad_enabled(False)

    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.nn.functional.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


def print_gpu_info():
    """
    Print detailed GPU information if available.
    """
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("GPU not available")


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")