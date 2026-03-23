"""Core utilities for low-power computer vision."""

import logging
import random
import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import tensorflow as tf
from omegaconf import DictConfig


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # TensorFlow settings for reproducibility
    tf.config.experimental.enable_op_determinism()


def get_device(device_type: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device_type: Device type ('cuda', 'cpu', 'mps', or None for auto)
        
    Returns:
        PyTorch device object
    """
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    device = torch.device(device_type)
    logging.info(f"Using device: {device}")
    return device


def get_tf_device(device_type: Optional[str] = None) -> str:
    """Get the appropriate TensorFlow device.
    
    Args:
        device_type: Device type ('gpu', 'cpu', or None for auto)
        
    Returns:
        TensorFlow device string
    """
    if device_type is None:
        if tf.config.list_physical_devices('GPU'):
            device_type = "gpu"
        else:
            device_type = "cpu"
    
    logging.info(f"Using TensorFlow device: {device_type}")
    return device_type


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    from omegaconf import OmegaConf
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    from omegaconf import OmegaConf
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(config, output_path)


def format_model_size(size_bytes: int) -> str:
    """Format model size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def calculate_model_complexity(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter count, trainable parameters, and model size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in bytes (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = param_size + buffer_size
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_bytes": model_size,
        "model_size_mb": model_size / (1024 * 1024)
    }


def validate_input_shape(input_shape: Tuple[int, ...], expected_shape: Tuple[int, ...]) -> bool:
    """Validate input shape against expected shape.
    
    Args:
        input_shape: Actual input shape
        expected_shape: Expected input shape
        
    Returns:
        True if shapes are compatible
    """
    if len(input_shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(input_shape, expected_shape):
        if expected != -1 and actual != expected:
            return False
    
    return True


class PerformanceProfiler:
    """Simple performance profiler for edge inference."""
    
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        import time
        self.times[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        import time
        if f"{name}_start" not in self.times:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.time() - self.times[f"{name}_start"]
        self.times[name] = duration
        self.counts[name] = self.counts.get(name, 0) + 1
        
        return duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        for name, duration in self.times.items():
            if not name.endswith("_start"):
                count = self.counts.get(name, 1)
                stats[name] = {
                    "total_time": duration,
                    "avg_time": duration / count,
                    "count": count
                }
        return stats
    
    def reset(self) -> None:
        """Reset all timers."""
        self.times.clear()
        self.counts.clear()
