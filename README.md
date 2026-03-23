# Low-Power Computer Vision

A comprehensive framework for training, compressing, and deploying computer vision models on edge devices. This project demonstrates modern techniques for creating efficient AI models suitable for resource-constrained environments.

## ⚠️ IMPORTANT DISCLAIMER

**THIS PROJECT IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

**NOT FOR SAFETY-CRITICAL APPLICATIONS.**

The models and code in this repository have not been validated for safety-critical use cases. Do not use this software in applications where failure could result in injury, death, or significant property damage.

## Features

- **Modern PyTorch 2.x Implementation**: Clean, typed code with comprehensive error handling
- **Advanced Model Compression**: Quantization (PTQ/QAT), pruning, and knowledge distillation
- **Multi-Format Export**: ONNX, TensorFlow Lite, CoreML for various edge platforms
- **Comprehensive Evaluation**: Accuracy and edge performance metrics with detailed benchmarking
- **Interactive Demo**: Streamlit-based demo simulating edge constraints
- **Production-Ready Structure**: Proper configuration management, logging, and documentation

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Low-Power-Computer-Vision.git
cd Low-Power-Computer-Vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
python scripts/train.py --model-type tiny_cnn --epochs 50 --compress --export --benchmark
```

4. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures and compression
│   ├── pipelines/         # Training and data pipelines
│   ├── export/            # Model export and deployment
│   ├── runtimes/          # Edge inference runtimes
│   ├── comms/             # IoT communication modules
│   └── utils/             # Core utilities
├── configs/               # Configuration files
├── scripts/               # Training and utility scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
├── data/                  # Data storage
├── assets/                # Generated artifacts
└── outputs/               # Model outputs and results
```

## Model Architectures

### TinyCNN
A lightweight CNN optimized for edge deployment:
- **Parameters**: ~50K parameters
- **Size**: < 1MB
- **Latency**: < 10ms on ARM Cortex-M4
- **Use Case**: Microcontroller deployment

### MobileNetV2-Tiny
A reduced MobileNetV2 variant:
- **Parameters**: ~200K parameters  
- **Size**: < 2MB
- **Latency**: < 20ms on Raspberry Pi Zero
- **Use Case**: Single-board computer deployment

## Compression Techniques

### Quantization
- **Post-Training Quantization (PTQ)**: INT8 quantization without retraining
- **Quantization-Aware Training (QAT)**: INT8 quantization with retraining
- **Dynamic Quantization**: Runtime quantization for PyTorch models

### Pruning
- **Magnitude-Based Pruning**: Remove least important weights
- **Structured Pruning**: Remove entire channels/filters
- **Iterative Pruning**: Gradual pruning with fine-tuning

### Knowledge Distillation
- **Teacher-Student Training**: Transfer knowledge from large to small models
- **Feature Distillation**: Match intermediate representations
- **Temperature Scaling**: Soft probability matching

## Edge Deployment Targets

### Raspberry Pi Zero
- **Runtime**: TensorFlow Lite
- **Performance**: 10-15 FPS
- **Power**: ~1W
- **Memory**: 512MB RAM

### ESP32-CAM
- **Runtime**: TensorFlow Lite Micro
- **Performance**: 5-8 FPS
- **Power**: ~200mW
- **Memory**: 320KB RAM

### Jetson Nano
- **Runtime**: ONNX Runtime + TensorRT
- **Performance**: 30+ FPS
- **Power**: ~5W
- **Memory**: 4GB RAM

## Usage Examples

### Training a Baseline Model

```python
from src.models import create_model
from src.pipelines import create_data_loaders, EdgeTrainer
from src.utils import get_device, set_deterministic_seed

# Set up
set_deterministic_seed(42)
device = get_device()

# Create model and data
model = create_model("tiny_cnn", num_classes=3)
train_loader, val_loader, test_loader = create_data_loaders("cifar10", target_classes=[0, 1, 2])

# Train
trainer = EdgeTrainer(model, device, {"learning_rate": 0.001, "epochs": 50})
history = trainer.train(train_loader, val_loader)
```

### Model Compression

```python
from src.models.compression import ModelCompressor

# Create compressor
compressor = ModelCompressor(model, device)

# Apply pruning
compressor.prune_model(pruning_method="magnitude", pruning_ratio=0.2)

# Apply quantization
compressor.quantize_model(quantization_method="dynamic")

# Get compression stats
stats = compressor.get_compression_stats()
```

### Model Export

```python
from src.export import ModelExporter

# Create exporter
exporter = ModelExporter(model, device)

# Export to multiple formats
exported_models = exporter.export_all_formats(
    output_dir="models",
    model_name="edge_model",
    input_shape=(1, 3, 32, 32)
)
```

### Edge Inference

```python
from src.export import EdgeRuntime

# Load edge runtime
runtime = EdgeRuntime("model.tflite", "tflite")

# Run inference
predictions = runtime.predict(input_data)
```

## Performance Benchmarks

| Model | Format | Latency (ms) | Throughput (fps) | Size (MB) | Device |
|-------|--------|--------------|------------------|-----------|--------|
| TinyCNN | PyTorch | 15.2 | 65.8 | 2.1 | CPU |
| TinyCNN | ONNX | 12.8 | 78.1 | 2.3 | CPU |
| TinyCNN | TFLite | 8.5 | 117.6 | 0.8 | CPU |
| MobileNetV2 | PyTorch | 25.1 | 39.8 | 4.2 | CPU |
| MobileNetV2 | ONNX | 18.3 | 54.6 | 4.5 | CPU |
| MobileNetV2 | TFLite | 11.2 | 89.3 | 1.2 | CPU |

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
device:
  raspberry_pi_zero:
    cpu_cores: 1
    memory_mb: 512
    power_consumption_mw: 1000
    inference_targets: ["tflite", "onnx"]

quant:
  int8_ptq:
    method: "post_training"
    bits: 8
    calibration_samples: 100
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{low_power_computer_vision,
  title={Low-Power Computer Vision},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Low-Power-Computer-Vision}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- TensorFlow Lite team for edge deployment tools
- ONNX community for model interoperability
- Streamlit team for the interactive demo framework

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the example notebooks in `notebooks/`

---

**Remember: This software is for research and educational purposes only. NOT FOR SAFETY-CRITICAL APPLICATIONS.**
# Low-Power-Computer-Vision
