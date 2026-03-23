#!/usr/bin/env python3
"""Quick start script for low-power computer vision demo.

This script provides a simple way to get started with the project
without running the full training pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import setup_logging, set_deterministic_seed, get_device
from models import create_model, get_model_summary
from models.compression import ModelCompressor
from export import ModelExporter


def create_demo_model():
    """Create a demo model for quick testing."""
    print("Creating demo model...")
    
    # Set up
    set_deterministic_seed(42)
    device = get_device()
    
    # Create model
    model = create_model("tiny_cnn", num_classes=3)
    model.to(device)
    
    print(f"Model created: {get_model_summary(model)}")
    
    return model, device


def test_inference(model, device):
    """Test model inference."""
    print("Testing inference...")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        predictions = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    class_names = ["Airplane", "Car", "Bird"]
    
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    return predictions


def test_compression(model, device):
    """Test model compression."""
    print("Testing model compression...")
    
    # Create compressor
    compressor = ModelCompressor(model, device)
    
    # Apply pruning
    compressed_model = compressor.prune_model(
        pruning_method="magnitude",
        pruning_ratio=0.2
    )
    
    # Get compression stats
    stats = compressor.get_compression_stats()
    
    print(f"Compression stats:")
    print(f"  Parameter reduction: {stats.get('parameter_reduction', 0):.2%}")
    print(f"  Size reduction: {stats.get('size_reduction', 0):.2%}")
    
    return compressed_model, stats


def test_export(model, device, output_dir="demo_outputs"):
    """Test model export."""
    print("Testing model export...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create exporter
    exporter = ModelExporter(model, device)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "demo_model.onnx")
    try:
        exported_path = exporter.export_to_onnx(onnx_path)
        print(f"ONNX model exported to: {exported_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    return onnx_path


def create_demo_visualization(predictions, output_dir="demo_outputs"):
    """Create demo visualization."""
    print("Creating demo visualization...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ["Airplane", "Car", "Bird"]
    
    # Create bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, predictions[0].numpy())
    plt.title("Class Probabilities")
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    
    # Color bars
    for i, bar in enumerate(bars):
        bar.set_color('red' if i == predictions.argmax().item() else 'lightblue')
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, predictions[0].numpy())):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "demo_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {plot_path}")
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick start demo")
    parser.add_argument("--output-dir", default="demo_outputs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    print("🚀 Low-Power Computer Vision Quick Start Demo")
    print("=" * 50)
    
    try:
        # Create demo model
        model, device = create_demo_model()
        
        # Test inference
        predictions = test_inference(model, device)
        
        # Test compression
        compressed_model, compression_stats = test_compression(model, device)
        
        # Test export
        export_path = test_export(model, device, args.output_dir)
        
        # Create visualization
        create_demo_visualization(predictions, args.output_dir)
        
        print("\n✅ Demo completed successfully!")
        print(f"📁 Outputs saved to: {args.output_dir}")
        print("\n🔬 Next steps:")
        print("  1. Run full training: python scripts/train.py --compress --export")
        print("  2. Launch interactive demo: streamlit run demo/app.py")
        print("  3. Read the README.md for detailed documentation")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
