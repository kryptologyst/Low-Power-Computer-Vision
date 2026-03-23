"""Modern PyTorch models for low-power computer vision."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class TinyCNN(nn.Module):
    """Lightweight CNN for edge deployment.
    
    A minimal CNN architecture optimized for low-power devices with
    reduced parameters and computational complexity.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        input_size: int = 32,
        dropout_rate: float = 0.2,
    ):
        """Initialize TinyCNN model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            input_size: Input image size (assumed square)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """Get model information and complexity metrics.
        
        Returns:
            Dictionary with model metrics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_bytes = param_size + buffer_size
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_bytes / (1024 * 1024),
            "input_shape": (self.input_channels, self.input_size, self.input_size),
            "num_classes": self.num_classes,
        }


class MobileNetV2Tiny(nn.Module):
    """Tiny MobileNetV2 variant for edge deployment.
    
    A reduced version of MobileNetV2 with fewer layers and channels
    optimized for low-power devices.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        width_multiplier: float = 0.5,
        dropout_rate: float = 0.2,
    ):
        """Initialize MobileNetV2Tiny model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            width_multiplier: Width multiplier for channel scaling
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        
        # Calculate channel widths
        def make_divisible(v: int, divisor: int = 8) -> int:
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        input_channel = make_divisible(32 * width_multiplier)
        last_channel = make_divisible(1280 * width_multiplier)
        
        # Initial convolution
        self.features = [
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        
        # Inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel
        
        # Final convolution
        self.features.append(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False)
        )
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """Get model information and complexity metrics.
        
        Returns:
            Dictionary with model metrics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_bytes = param_size + buffer_size
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_bytes / (1024 * 1024),
            "width_multiplier": self.width_multiplier,
            "num_classes": self.num_classes,
        }


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2."""
    
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ):
        """Initialize inverted residual block.
        
        Args:
            inp: Input channels
            oup: Output channels
            stride: Stride for convolution
            expand_ratio: Expansion ratio
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride, 1, 
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def create_model(
    model_type: str = "tiny_cnn",
    num_classes: int = 3,
    **kwargs
) -> nn.Module:
    """Create a model instance.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model instance
    """
    if model_type == "tiny_cnn":
        return TinyCNN(num_classes=num_classes, **kwargs)
    elif model_type == "mobilenetv2_tiny":
        return MobileNetV2Tiny(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_summary(model: nn.Module) -> str:
    """Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"Model: {model.__class__.__name__}\n"
    summary += f"Total parameters: {total_params:,}\n"
    summary += f"Trainable parameters: {trainable_params:,}\n"
    
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        summary += f"Model size: {info['model_size_mb']:.2f} MB\n"
    
    return summary
