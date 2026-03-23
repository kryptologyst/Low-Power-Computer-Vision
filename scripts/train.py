#!/usr/bin/env python3
"""Main training script for low-power computer vision models.

This script demonstrates the complete pipeline for training, compressing,
and deploying edge computer vision models.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import (
    setup_logging,
    set_deterministic_seed,
    get_device,
    load_config,
    save_config,
    format_model_size,
    calculate_model_complexity,
    PerformanceProfiler,
)
from models import create_model, get_model_summary
from models.compression import ModelCompressor, KnowledgeDistillation, create_compressed_model
from pipelines import (
    create_data_loaders,
    EdgeTrainer,
    plot_training_history,
    plot_confusion_matrix,
)
from export import ModelExporter, benchmark_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and deploy low-power computer vision models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="tiny_cnn",
        choices=["tiny_cnn", "mobilenetv2_tiny"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps, or None for auto)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and results"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Apply model compression"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export models to edge formats"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark exported models"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def train_baseline_model(
    config: DictConfig,
    model_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    output_dir: str,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train baseline model.
    
    Args:
        config: Configuration object
        model_type: Type of model to train
        num_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device for computation
        output_dir: Output directory
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    logger = logging.getLogger(__name__)
    logger.info("Training baseline model")
    
    # Create model
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
    )
    
    logger.info(f"Model created: {get_model_summary(model)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="cifar10",
        target_classes=list(range(num_classes)),
        batch_size=batch_size,
        num_workers=2,
    )
    
    # Training configuration
    train_config = {
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "epochs": epochs,
    }
    
    # Create trainer
    trainer = EdgeTrainer(model, device, train_config)
    
    # Train model
    history = trainer.train(train_loader, val_loader, epochs)
    
    # Evaluate on test set
    test_results = trainer.evaluate(
        test_loader,
        class_names=[f"Class_{i}" for i in range(num_classes)]
    )
    
    logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "baseline_model.pth")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.npz")
    np.savez(history_path, **history)
    
    # Plot training history
    plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    # Plot confusion matrix
    if "confusion_matrix" in test_results:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            [f"Class_{i}" for i in range(num_classes)],
            cm_path
        )
    
    return model, history


def compress_model(
    model: nn.Module,
    device: torch.device,
    compression_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    output_dir: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Compress model using various techniques.
    
    Args:
        model: Baseline model
        device: Device for computation
        compression_config: Compression configuration
        train_loader: Training data loader for calibration
        output_dir: Output directory
        
    Returns:
        Tuple of (compressed_model, compression_stats)
    """
    logger = logging.getLogger(__name__)
    logger.info("Compressing model")
    
    # Create compressor
    compressor = ModelCompressor(model, device)
    
    # Apply pruning if specified
    if "pruning" in compression_config:
        pruning_config = compression_config["pruning"]
        compressor.prune_model(
            pruning_method=pruning_config.get("method", "magnitude"),
            pruning_ratio=pruning_config.get("ratio", 0.2),
            structured=pruning_config.get("structured", False),
        )
    
    # Apply quantization if specified
    if "quantization" in compression_config:
        quant_config = compression_config["quantization"]
        
        # Get calibration data
        calibration_data = []
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx >= 10:  # Use first 10 batches for calibration
                break
            calibration_data.append(data)
        
        calibration_data = torch.cat(calibration_data, dim=0)
        
        compressor.quantize_model(
            quantization_method=quant_config.get("method", "dynamic"),
            calibration_data=calibration_data,
        )
    
    # Get compression statistics
    stats = compressor.get_compression_stats()
    
    logger.info(f"Compression stats: {stats}")
    
    # Save compressed model
    compressed_model_path = os.path.join(output_dir, "compressed_model.pth")
    torch.save(compressor.model.state_dict(), compressed_model_path)
    
    return compressor.model, stats


def export_models(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    device: torch.device,
    output_dir: str,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
) -> Dict[str, str]:
    """Export models to edge formats.
    
    Args:
        baseline_model: Baseline PyTorch model
        compressed_model: Compressed PyTorch model
        device: Device for computation
        output_dir: Output directory
        input_shape: Input tensor shape
        
    Returns:
        Dictionary mapping model names to exported file paths
    """
    logger = logging.getLogger(__name__)
    logger.info("Exporting models to edge formats")
    
    exported_models = {}
    
    # Export baseline model
    baseline_exporter = ModelExporter(baseline_model, device)
    baseline_exports = baseline_exporter.export_all_formats(
        output_dir=os.path.join(output_dir, "baseline"),
        model_name="baseline",
        input_shape=input_shape,
    )
    exported_models.update({f"baseline_{k}": v for k, v in baseline_exports.items()})
    
    # Export compressed model
    compressed_exporter = ModelExporter(compressed_model, device)
    compressed_exports = compressed_exporter.export_all_formats(
        output_dir=os.path.join(output_dir, "compressed"),
        model_name="compressed",
        input_shape=input_shape,
        quantization_config={"method": "int8"},
    )
    exported_models.update({f"compressed_{k}": v for k, v in compressed_exports.items()})
    
    logger.info(f"Exported models: {list(exported_models.keys())}")
    
    return exported_models


def benchmark_models(
    exported_models: Dict[str, str],
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    num_iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Benchmark exported models.
    
    Args:
        exported_models: Dictionary of exported model paths
        input_shape: Input tensor shape
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    logger = logging.getLogger(__name__)
    logger.info("Benchmarking exported models")
    
    benchmark_results = {}
    
    for model_name, model_path in exported_models.items():
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        # Determine runtime type from file extension
        if model_path.endswith(".onnx"):
            runtime_type = "onnx"
        elif model_path.endswith(".tflite"):
            runtime_type = "tflite"
        elif model_path.endswith(".mlmodel"):
            runtime_type = "coreml"
        else:
            logger.warning(f"Unknown model format: {model_path}")
            continue
        
        try:
            results = benchmark_model(
                model_path=model_path,
                runtime_type=runtime_type,
                input_shape=input_shape,
                num_iterations=num_iterations,
            )
            benchmark_results[model_name] = results
            
            logger.info(
                f"{model_name}: "
                f"Latency: {results['mean_latency_ms']:.2f}ms, "
                f"Throughput: {results['throughput_fps']:.2f}fps, "
                f"Size: {results['model_size_mb']:.2f}MB"
            )
            
        except Exception as e:
            logger.error(f"Benchmarking failed for {model_name}: {e}")
    
    return benchmark_results


def create_performance_report(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    compression_stats: Dict[str, Any],
    benchmark_results: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    """Create comprehensive performance report.
    
    Args:
        baseline_model: Baseline model
        compressed_model: Compressed model
        compression_stats: Compression statistics
        benchmark_results: Benchmark results
        output_dir: Output directory
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating performance report")
    
    # Calculate model complexities
    baseline_complexity = calculate_model_complexity(baseline_model)
    compressed_complexity = calculate_model_complexity(compressed_model)
    
    # Create report
    report_path = os.path.join(output_dir, "performance_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Low-Power Computer Vision Performance Report\n\n")
        f.write("## Model Complexity\n\n")
        f.write("### Baseline Model\n")
        f.write(f"- Total Parameters: {baseline_complexity['total_parameters']:,}\n")
        f.write(f"- Trainable Parameters: {baseline_complexity['trainable_parameters']:,}\n")
        f.write(f"- Model Size: {baseline_complexity['model_size_mb']:.2f} MB\n\n")
        
        f.write("### Compressed Model\n")
        f.write(f"- Total Parameters: {compressed_complexity['total_parameters']:,}\n")
        f.write(f"- Trainable Parameters: {compressed_complexity['trainable_parameters']:,}\n")
        f.write(f"- Model Size: {compressed_complexity['model_size_mb']:.2f} MB\n\n")
        
        f.write("## Compression Statistics\n\n")
        if "parameter_reduction" in compression_stats:
            f.write(f"- Parameter Reduction: {compression_stats['parameter_reduction']:.2%}\n")
        if "size_reduction" in compression_stats:
            f.write(f"- Size Reduction: {compression_stats['size_reduction']:.2%}\n")
        f.write(f"- Compression History: {compression_stats.get('compression_history', [])}\n\n")
        
        f.write("## Benchmark Results\n\n")
        f.write("| Model | Format | Latency (ms) | Throughput (fps) | Size (MB) |\n")
        f.write("|-------|--------|--------------|------------------|----------|\n")
        
        for model_name, results in benchmark_results.items():
            f.write(
                f"| {model_name} | {model_name.split('_')[-1]} | "
                f"{results['mean_latency_ms']:.2f} | "
                f"{results['throughput_fps']:.2f} | "
                f"{results['model_size_mb']:.2f} |\n"
            )
        
        f.write("\n## Edge Deployment Recommendations\n\n")
        f.write("Based on the results above, consider the following for edge deployment:\n\n")
        f.write("- **Raspberry Pi Zero**: Use compressed TFLite model for best performance\n")
        f.write("- **ESP32-CAM**: Use compressed TFLite model with further quantization\n")
        f.write("- **Jetson Nano**: Use ONNX model for GPU acceleration\n")
        f.write("- **Mobile Devices**: Use CoreML model for iOS deployment\n\n")
        
        f.write("## Disclaimer\n\n")
        f.write("**This model is for research and educational purposes only.**\n")
        f.write("**NOT FOR SAFETY-CRITICAL APPLICATIONS.**\n")
        f.write("The model has not been validated for safety-critical use cases.\n")
    
    logger.info(f"Performance report saved to: {report_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting low-power computer vision training pipeline")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Train baseline model
        baseline_model, training_history = train_baseline_model(
            config=config,
            model_type=args.model_type,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            output_dir=args.output_dir,
        )
        
        compressed_model = None
        compression_stats = {}
        
        # Compress model if requested
        if args.compress:
            # Get training data for compression
            train_loader, _, _ = create_data_loaders(
                dataset_name="cifar10",
                target_classes=list(range(args.num_classes)),
                batch_size=args.batch_size,
                num_workers=2,
            )
            
            compression_config = {
                "pruning": {"method": "magnitude", "ratio": 0.2},
                "quantization": {"method": "dynamic"},
            }
            
            compressed_model, compression_stats = compress_model(
                baseline_model,
                device,
                compression_config,
                train_loader,
                args.output_dir,
            )
        
        # Export models if requested
        exported_models = {}
        if args.export:
            if compressed_model is None:
                compressed_model = baseline_model
            
            exported_models = export_models(
                baseline_model,
                compressed_model,
                device,
                args.output_dir,
            )
        
        # Benchmark models if requested
        benchmark_results = {}
        if args.benchmark and exported_models:
            benchmark_results = benchmark_models(exported_models)
        
        # Create performance report
        if compressed_model is not None and benchmark_results:
            create_performance_report(
                baseline_model,
                compressed_model,
                compression_stats,
                benchmark_results,
                args.output_dir,
            )
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
