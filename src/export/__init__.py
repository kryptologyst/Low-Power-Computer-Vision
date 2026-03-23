"""Model export and deployment utilities for edge devices."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import numpy as np
import onnx
import tensorflow as tf
from pathlib import Path


class ModelExporter:
    """Export PyTorch models to various edge formats."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize model exporter.
        
        Args:
            model: PyTorch model to export
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            output_path: Output file path
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to exported ONNX model
        """
        logging.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logging.info("ONNX export successful")
        return output_path
    
    def export_to_tflite(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 32, 32, 3),
        quantization: str = "int8",
        representative_dataset: Optional[np.ndarray] = None,
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            output_path: Output file path
            input_shape: Input tensor shape (NHWC format)
            quantization: Quantization type ('float32', 'int8', 'int16')
            representative_dataset: Dataset for quantization calibration
            
        Returns:
            Path to exported TFLite model
        """
        logging.info(f"Exporting model to TFLite: {output_path}")
        
        # First export to ONNX
        onnx_path = output_path.replace(".tflite", ".onnx")
        self.export_to_onnx(onnx_path, input_shape=(1, 3, 32, 32))
        
        # Convert ONNX to TensorFlow
        try:
            import onnx_tf
            tf_rep = onnx_tf.backend.prepare(onnx.load(onnx_path))
            tf_model = tf_rep.export_graph()
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model])
            
            if quantization == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if representative_dataset is not None:
                    def representative_data_gen():
                        for i in range(min(100, len(representative_dataset))):
                            yield [representative_dataset[i:i+1]]
                    converter.representative_dataset = representative_data_gen
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            
            logging.info("TFLite export successful")
            return output_path
            
        except ImportError:
            logging.warning("onnx-tf not available, skipping TFLite export")
            return ""
    
    def export_to_coreml(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    ) -> str:
        """Export model to CoreML format.
        
        Args:
            output_path: Output file path
            input_shape: Input tensor shape
            
        Returns:
            Path to exported CoreML model
        """
        logging.info(f"Exporting model to CoreML: {output_path}")
        
        try:
            import coremltools as ct
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape, name="input")],
                outputs=[ct.TensorType(name="output")],
            )
            
            # Save CoreML model
            coreml_model.save(output_path)
            
            logging.info("CoreML export successful")
            return output_path
            
        except ImportError:
            logging.warning("coremltools not available, skipping CoreML export")
            return ""
    
    def export_all_formats(
        self,
        output_dir: str,
        model_name: str = "model",
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        quantization_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            output_dir: Output directory
            model_name: Base name for exported models
            input_shape: Input tensor shape
            quantization_config: Quantization configuration
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_models = {}
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        exported_models["onnx"] = self.export_to_onnx(onnx_path, input_shape)
        
        # Export to TFLite
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        if quantization_config:
            exported_models["tflite"] = self.export_to_tflite(
                tflite_path,
                input_shape=(1, input_shape[2], input_shape[3], input_shape[1]),  # NHWC
                quantization=quantization_config.get("method", "int8"),
                representative_dataset=quantization_config.get("calibration_data"),
            )
        else:
            exported_models["tflite"] = self.export_to_tflite(tflite_path, input_shape)
        
        # Export to CoreML
        coreml_path = os.path.join(output_dir, f"{model_name}.mlmodel")
        exported_models["coreml"] = self.export_to_coreml(coreml_path, input_shape)
        
        return exported_models


class EdgeRuntime:
    """Runtime for edge model inference."""
    
    def __init__(self, model_path: str, runtime_type: str = "onnx"):
        """Initialize edge runtime.
        
        Args:
            model_path: Path to model file
            runtime_type: Type of runtime ('onnx', 'tflite', 'coreml')
        """
        self.model_path = model_path
        self.runtime_type = runtime_type.lower()
        self.session = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model based on runtime type."""
        if self.runtime_type == "onnx":
            self._load_onnx_model()
        elif self.runtime_type == "tflite":
            self._load_tflite_model()
        elif self.runtime_type == "coreml":
            self._load_coreml_model()
        else:
            raise ValueError(f"Unknown runtime type: {self.runtime_type}")
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            providers = ["CPUExecutionProvider"]
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
            
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            
            logging.info("ONNX model loaded successfully")
            
        except ImportError:
            raise ImportError("onnxruntime not available")
    
    def _load_tflite_model(self) -> None:
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            
            self.session = tflite.Interpreter(model_path=self.model_path)
            self.session.allocate_tensors()
            
            logging.info("TFLite model loaded successfully")
            
        except ImportError:
            raise ImportError("tflite_runtime not available")
    
    def _load_coreml_model(self) -> None:
        """Load CoreML model."""
        try:
            import coremltools as ct
            
            self.session = ct.models.MLModel(self.model_path)
            
            logging.info("CoreML model loaded successfully")
            
        except ImportError:
            raise ImportError("coremltools not available")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input data array
            
        Returns:
            Prediction results
        """
        if self.runtime_type == "onnx":
            return self._predict_onnx(input_data)
        elif self.runtime_type == "tflite":
            return self._predict_tflite(input_data)
        elif self.runtime_type == "coreml":
            return self._predict_coreml(input_data)
        else:
            raise ValueError(f"Unknown runtime type: {self.runtime_type}")
    
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        result = self.session.run([output_name], {input_name: input_data})
        return result[0]
    
    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """TFLite inference."""
        input_details = self.session.get_input_details()
        output_details = self.session.get_output_details()
        
        # Set input
        self.session.set_tensor(input_details[0]["index"], input_data)
        
        # Run inference
        self.session.invoke()
        
        # Get output
        output_data = self.session.get_tensor(output_details[0]["index"])
        return output_data
    
    def _predict_coreml(self, input_data: np.ndarray) -> np.ndarray:
        """CoreML inference."""
        # Convert numpy array to PIL Image for CoreML
        from PIL import Image
        
        if input_data.shape[1] == 3:  # CHW format
            input_data = np.transpose(input_data, (1, 2, 0))  # HWC format
        
        # Normalize to 0-255 range
        if input_data.max() <= 1.0:
            input_data = (input_data * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(input_data)
        
        # Run inference
        result = self.session.predict({"input": image})
        
        # Extract output
        output = result["output"]
        if isinstance(output, np.ndarray):
            return output
        else:
            return np.array(output)


def benchmark_model(
    model_path: str,
    runtime_type: str,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark model inference performance.
    
    Args:
        model_path: Path to model file
        runtime_type: Type of runtime
        input_shape: Input tensor shape
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with performance metrics
    """
    import time
    
    # Initialize runtime
    runtime = EdgeRuntime(model_path, runtime_type)
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup_iterations):
        runtime.predict(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        runtime.predict(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    
    return {
        "mean_latency_ms": np.mean(times) * 1000,
        "std_latency_ms": np.std(times) * 1000,
        "p50_latency_ms": np.percentile(times, 50) * 1000,
        "p95_latency_ms": np.percentile(times, 95) * 1000,
        "p99_latency_ms": np.percentile(times, 99) * 1000,
        "throughput_fps": 1.0 / np.mean(times),
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    }
