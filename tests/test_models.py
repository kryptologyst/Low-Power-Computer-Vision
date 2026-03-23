"""Tests for low-power computer vision models."""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    set_deterministic_seed,
    get_device,
    format_model_size,
    calculate_model_complexity,
    PerformanceProfiler,
)
from models import create_model, get_model_summary
from models.compression import ModelCompressor
from pipelines import EdgeVisionDataset


class TestUtils:
    """Test utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        
        set_deterministic_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_format_model_size(self):
        """Test model size formatting."""
        assert format_model_size(1024) == "1024.00 B"
        assert format_model_size(1024 * 1024) == "1.00 MB"
        assert format_model_size(1024 * 1024 * 1024) == "1.00 GB"
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        profiler = PerformanceProfiler()
        
        profiler.start_timer("test")
        import time
        time.sleep(0.01)  # Small delay
        duration = profiler.end_timer("test")
        
        assert duration > 0
        assert "test" in profiler.get_stats()


class TestModels:
    """Test model architectures."""
    
    def test_tiny_cnn_creation(self):
        """Test TinyCNN model creation."""
        model = create_model("tiny_cnn", num_classes=3)
        
        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 3
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (1, 3)
    
    def test_mobilenetv2_tiny_creation(self):
        """Test MobileNetV2Tiny model creation."""
        model = create_model("mobilenetv2_tiny", num_classes=3)
        
        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 3
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (1, 3)
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = create_model("tiny_cnn", num_classes=3)
        summary = get_model_summary(model)
        
        assert isinstance(summary, str)
        assert "TinyCNN" in summary
        assert "parameters" in summary
    
    def test_model_complexity(self):
        """Test model complexity calculation."""
        model = create_model("tiny_cnn", num_classes=3)
        complexity = calculate_model_complexity(model)
        
        assert "total_parameters" in complexity
        assert "trainable_parameters" in complexity
        assert "model_size_bytes" in complexity
        assert complexity["total_parameters"] > 0


class TestCompression:
    """Test model compression techniques."""
    
    def test_model_compressor_creation(self):
        """Test ModelCompressor creation."""
        model = create_model("tiny_cnn", num_classes=3)
        device = get_device()
        
        compressor = ModelCompressor(model, device)
        
        assert compressor.model == model
        assert compressor.device == device
    
    def test_pruning(self):
        """Test model pruning."""
        model = create_model("tiny_cnn", num_classes=3)
        device = get_device()
        
        compressor = ModelCompressor(model, device)
        
        # Test magnitude pruning
        compressed_model = compressor.prune_model(
            pruning_method="magnitude",
            pruning_ratio=0.1
        )
        
        assert isinstance(compressed_model, torch.nn.Module)
        
        # Test compression stats
        stats = compressor.get_compression_stats()
        assert "original_parameters" in stats
        assert "compressed_parameters" in stats
    
    def test_quantization(self):
        """Test model quantization."""
        model = create_model("tiny_cnn", num_classes=3)
        device = get_device()
        
        compressor = ModelCompressor(model, device)
        
        # Test dynamic quantization
        compressed_model = compressor.quantize_model(
            quantization_method="dynamic"
        )
        
        assert isinstance(compressed_model, torch.nn.Module)


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_edge_vision_dataset(self):
        """Test EdgeVisionDataset."""
        # Create dummy data
        images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 3, 100)
        
        dataset = EdgeVisionDataset(images, labels)
        
        assert len(dataset) == 100
        
        # Test getting item
        image, label = dataset[0]
        assert isinstance(image, np.ndarray)
        assert isinstance(label, int)
        assert 0 <= label < 3
    
    def test_dataset_with_target_classes(self):
        """Test dataset with target classes filtering."""
        # Create dummy data
        images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 5, 100)  # 5 classes
        
        # Filter to only classes 0, 1, 2
        dataset = EdgeVisionDataset(images, labels, target_classes=[0, 1, 2])
        
        # Should have fewer items after filtering
        assert len(dataset) <= 100
        
        # All labels should be 0, 1, or 2
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert label in [0, 1, 2]


class TestIntegration:
    """Integration tests."""
    
    def test_training_pipeline(self):
        """Test basic training pipeline."""
        # Create model
        model = create_model("tiny_cnn", num_classes=3)
        device = get_device()
        
        # Create dummy data
        images = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 3, 50)
        
        dataset = EdgeVisionDataset(images, labels)
        
        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 2:  # Test only first 2 batches
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            assert loss.item() >= 0
    
    def test_model_export(self):
        """Test model export functionality."""
        model = create_model("tiny_cnn", num_classes=3)
        device = get_device()
        
        # Test ONNX export
        with tempfile.TemporaryDirectory() as temp_dir:
            from export import ModelExporter
            
            exporter = ModelExporter(model, device)
            
            onnx_path = os.path.join(temp_dir, "test_model.onnx")
            exported_path = exporter.export_to_onnx(onnx_path)
            
            assert os.path.exists(exported_path)
            assert exported_path.endswith(".onnx")


if __name__ == "__main__":
    pytest.main([__file__])
