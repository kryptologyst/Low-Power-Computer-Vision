# Migration Note

This file has been replaced by the modernized low-power computer vision framework.

## What's New

The project has been completely refactored and modernized with:

- **Modern PyTorch 2.x Implementation**: Clean, typed code with comprehensive error handling
- **Advanced Model Compression**: Quantization (PTQ/QAT), pruning, and knowledge distillation  
- **Multi-Format Export**: ONNX, TensorFlow Lite, CoreML for various edge platforms
- **Comprehensive Evaluation**: Accuracy and edge performance metrics with detailed benchmarking
- **Interactive Demo**: Streamlit-based demo simulating edge constraints
- **Production-Ready Structure**: Proper configuration management, logging, and documentation

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run quick demo**:
   ```bash
   python scripts/quick_start.py
   ```

3. **Full training pipeline**:
   ```bash
   python scripts/train.py --model-type tiny_cnn --epochs 50 --compress --export --benchmark
   ```

4. **Interactive demo**:
   ```bash
   streamlit run demo/app.py
   ```

## Key Improvements

- ✅ **Deterministic seeding** for reproducible results
- ✅ **Type hints and docstrings** for better code quality
- ✅ **Device fallback** (CUDA → CPU) for compatibility
- ✅ **Advanced compression** with pruning and quantization
- ✅ **Multi-target export** (ONNX, TFLite, CoreML)
- ✅ **Comprehensive benchmarking** with performance metrics
- ✅ **Interactive demo** with real-time visualization
- ✅ **Safety scaffold** with privacy considerations
- ✅ **Production-ready structure** with proper testing

## Original Implementation

The original TensorFlow implementation has been preserved for reference:

```python
# Original TensorFlow implementation (simplified)
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Filter to 3 classes
selected_classes = [0, 1, 2]
train_mask = np.isin(y_train, selected_classes).flatten()
test_mask = np.isin(y_test, selected_classes).flatten()

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Build model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {acc:.4f}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("tiny_cnn_quantized.tflite", "wb") as f:
    f.write(tflite_model)
```

## Migration Guide

To migrate from the original implementation:

1. **Replace TensorFlow with PyTorch**: Use the new model architectures in `src/models/`
2. **Use modern data pipeline**: Leverage `src/pipelines/training.py` for data handling
3. **Apply compression**: Use `src/models/compression.py` for model optimization
4. **Export models**: Use `src/export/deployment.py` for multi-format export
5. **Run evaluation**: Use the comprehensive evaluation in `scripts/train.py`

## Documentation

- **README.md**: Complete project documentation
- **DISCLAIMER.md**: Safety and privacy considerations
- **demo/app.py**: Interactive Streamlit demo
- **scripts/**: Training and utility scripts
- **tests/**: Comprehensive test suite

---

**⚠️ DISCLAIMER: This software is for research and educational purposes only. NOT FOR SAFETY-CRITICAL APPLICATIONS.**