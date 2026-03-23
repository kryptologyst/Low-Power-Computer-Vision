"""Model compression and optimization techniques."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, quantize_static
import numpy as np


class ModelCompressor:
    """Unified model compression interface."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize model compressor.
        
        Args:
            model: PyTorch model to compress
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.original_model = None
        self.compression_history = []
    
    def prune_model(
        self,
        pruning_method: str = "magnitude",
        pruning_ratio: float = 0.2,
        structured: bool = False,
    ) -> nn.Module:
        """Apply pruning to the model.
        
        Args:
            pruning_method: Pruning method ('magnitude', 'random', 'gradient')
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        logging.info(f"Applying {pruning_method} pruning with ratio {pruning_ratio}")
        
        # Save original model
        if self.original_model is None:
            self.original_model = self._copy_model()
        
        if structured:
            return self._structured_pruning(pruning_ratio)
        else:
            return self._unstructured_pruning(pruning_method, pruning_ratio)
    
    def _unstructured_pruning(
        self, 
        method: str, 
        ratio: float
    ) -> nn.Module:
        """Apply unstructured pruning."""
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if method == "magnitude":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=ratio,
            )
        elif method == "random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=ratio,
            )
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.compression_history.append({
            "method": "unstructured_pruning",
            "ratio": ratio,
            "pruning_method": method,
        })
        
        return self.model
    
    def _structured_pruning(self, ratio: float) -> nn.Module:
        """Apply structured pruning."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated structured pruning
        logging.warning("Structured pruning implementation is simplified")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Simple channel pruning
                num_channels = module.out_channels
                channels_to_keep = int(num_channels * (1 - ratio))
                
                if channels_to_keep < num_channels:
                    # Create new module with fewer channels
                    new_module = nn.Conv2d(
                        module.in_channels,
                        channels_to_keep,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        bias=module.bias is not None,
                    )
                    
                    # Copy weights (simplified)
                    with torch.no_grad():
                        new_module.weight.data = module.weight.data[:channels_to_keep]
                        if module.bias is not None:
                            new_module.bias.data = module.bias.data[:channels_to_keep]
                    
                    # Replace module
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent_module = self.model.get_submodule(parent_name)
                        setattr(parent_module, name.split('.')[-1], new_module)
                    else:
                        setattr(self.model, name, new_module)
        
        self.compression_history.append({
            "method": "structured_pruning",
            "ratio": ratio,
        })
        
        return self.model
    
    def quantize_model(
        self,
        quantization_method: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Apply quantization to the model.
        
        Args:
            quantization_method: Quantization method ('dynamic', 'static')
            calibration_data: Data for static quantization calibration
            
        Returns:
            Quantized model
        """
        logging.info(f"Applying {quantization_method} quantization")
        
        # Save original model
        if self.original_model is None:
            self.original_model = self._copy_model()
        
        if quantization_method == "dynamic":
            return self._dynamic_quantization()
        elif quantization_method == "static":
            return self._static_quantization(calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {quantization_method}")
    
    def _dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization."""
        self.model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        self.compression_history.append({
            "method": "dynamic_quantization",
        })
        
        return self.model
    
    def _static_quantization(self, calibration_data: Optional[torch.Tensor]) -> nn.Module:
        """Apply static quantization."""
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare model for quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with sample data
        with torch.no_grad():
            for i in range(min(100, len(calibration_data))):
                sample = calibration_data[i:i+1].to(self.device)
                prepared_model(sample)
        
        # Convert to quantized model
        self.model = torch.quantization.convert(prepared_model)
        
        self.compression_history.append({
            "method": "static_quantization",
            "calibration_samples": len(calibration_data),
        })
        
        return self.model
    
    def _copy_model(self) -> nn.Module:
        """Create a copy of the model."""
        import copy
        return copy.deepcopy(self.model)
    
    def get_compression_stats(self) -> Dict[str, Union[int, float]]:
        """Get compression statistics."""
        if self.original_model is None:
            return {"error": "No original model saved"}
        
        original_params = sum(p.numel() for p in self.original_model.parameters())
        compressed_params = sum(p.numel() for p in self.model.parameters())
        
        original_size = sum(
            p.numel() * p.element_size() for p in self.original_model.parameters()
        )
        compressed_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        
        return {
            "original_parameters": original_params,
            "compressed_parameters": compressed_params,
            "parameter_reduction": (original_params - compressed_params) / original_params,
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "size_reduction": (original_size - compressed_size) / original_size,
            "compression_history": self.compression_history,
        }


class KnowledgeDistillation:
    """Knowledge distillation for model compression."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        temperature: float = 3.0,
        alpha: float = 0.7,
    ):
        """Initialize knowledge distillation.
        
        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            device: Device for computation
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # Set models to evaluation mode
        self.teacher_model.eval()
        self.student_model.train()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth targets
            
        Returns:
            Combined distillation loss
        """
        # Softmax with temperature
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            student_soft.log(),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (cross entropy)
        student_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            inputs: Input batch
            targets: Target batch
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss values
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        student_logits = self.student_model(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(student_logits, dim=1)
            accuracy = (predictions == targets).float().mean().item()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }


def create_compressed_model(
    model: nn.Module,
    compression_config: Dict[str, Union[str, float]],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Union[int, float]]]:
    """Create a compressed model based on configuration.
    
    Args:
        model: Original model
        compression_config: Compression configuration
        device: Device for computation
        
    Returns:
        Tuple of (compressed_model, compression_stats)
    """
    compressor = ModelCompressor(model, device)
    
    # Apply compression techniques
    if "pruning" in compression_config:
        pruning_config = compression_config["pruning"]
        compressor.prune_model(
            pruning_method=pruning_config.get("method", "magnitude"),
            pruning_ratio=pruning_config.get("ratio", 0.2),
            structured=pruning_config.get("structured", False),
        )
    
    if "quantization" in compression_config:
        quant_config = compression_config["quantization"]
        compressor.quantize_model(
            quantization_method=quant_config.get("method", "dynamic"),
            calibration_data=quant_config.get("calibration_data"),
        )
    
    stats = compressor.get_compression_stats()
    return compressor.model, stats
