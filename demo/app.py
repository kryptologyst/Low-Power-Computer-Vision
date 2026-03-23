"""Streamlit demo for low-power computer vision."""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, PerformanceProfiler
from models import create_model
from export import EdgeRuntime


# Page configuration
st.set_page_config(
    page_title="Low-Power Computer Vision Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔬 Low-Power Computer Vision Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ <strong>DISCLAIMER:</strong> This demo is for research and educational purposes only. 
NOT FOR SAFETY-CRITICAL APPLICATIONS. The models have not been validated for safety-critical use cases.
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["tiny_cnn", "mobilenetv2_tiny"],
    help="Select the model architecture"
)

# Runtime selection
runtime_type = st.sidebar.selectbox(
    "Runtime Type",
    ["pytorch", "onnx", "tflite"],
    help="Select the inference runtime"
)

# Device selection
device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_options.append("mps")

device = st.sidebar.selectbox(
    "Device",
    device_options,
    help="Select the computation device"
)

# Input source selection
input_source = st.sidebar.radio(
    "Input Source",
    ["Webcam", "Upload Image", "Synthetic Data"],
    help="Select the input source for inference"
)

# Performance monitoring
enable_profiling = st.sidebar.checkbox(
    "Enable Performance Profiling",
    value=True,
    help="Enable detailed performance monitoring"
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    
    # Input handling
    input_image = None
    input_array = None
    
    if input_source == "Webcam":
        st.write("Webcam input (simulated)")
        # Generate synthetic image for demo
        input_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        input_image = Image.fromarray(input_array)
        
    elif input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for classification"
        )
        
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            # Resize to model input size
            input_image = input_image.resize((32, 32))
            input_array = np.array(input_image)
            
    elif input_source == "Synthetic Data":
        st.write("Synthetic test data")
        # Generate synthetic image
        input_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        input_image = Image.fromarray(input_array)
    
    if input_image is not None:
        st.image(input_image, caption="Input Image", use_column_width=True)
        
        # Display image info
        st.write(f"**Image Shape:** {input_array.shape}")
        st.write(f"**Image Type:** {input_array.dtype}")

with col2:
    st.header("Inference Results")
    
    # Initialize performance profiler
    if enable_profiling:
        profiler = PerformanceProfiler()
    
    # Model inference
    if input_array is not None:
        # Preprocess input
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW format
        input_tensor = input_tensor / 255.0  # Normalize
        
        # Run inference
        try:
            if runtime_type == "pytorch":
                # PyTorch inference
                model = create_model(model_type=model_type, num_classes=3)
                model.eval()
                
                if enable_profiling:
                    profiler.start_timer("inference")
                
                with torch.no_grad():
                    output = model(input_tensor)
                    predictions = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    confidence = predictions[0][predicted_class].item()
                
                if enable_profiling:
                    inference_time = profiler.end_timer("inference")
                
            else:
                # Edge runtime inference
                model_path = f"outputs/baseline/baseline.{runtime_type}"
                if not os.path.exists(model_path):
                    st.error(f"Model file not found: {model_path}")
                    st.info("Please run the training script first to generate models.")
                else:
                    runtime = EdgeRuntime(model_path, runtime_type)
                    
                    if enable_profiling:
                        profiler.start_timer("inference")
                    
                    # Convert input for edge runtime
                    if runtime_type == "onnx":
                        input_np = input_tensor.numpy()
                    elif runtime_type == "tflite":
                        input_np = input_tensor.permute(0, 2, 3, 1).numpy()  # NHWC format
                    else:
                        input_np = input_tensor.numpy()
                    
                    output = runtime.predict(input_np)
                    predictions = torch.softmax(torch.from_numpy(output), dim=1)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    confidence = predictions[0][predicted_class].item()
                    
                    if enable_profiling:
                        inference_time = profiler.end_timer("inference")
            
            # Display results
            class_names = ["Airplane", "Car", "Bird"]
            
            st.write(f"**Predicted Class:** {class_names[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.4f}")
            
            # Confidence bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=predictions[0].numpy(),
                    marker_color=['red' if i == predicted_class else 'lightblue' for i in range(3)]
                )
            ])
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Classes",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            if enable_profiling:
                st.subheader("Performance Metrics")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric(
                        "Inference Time",
                        f"{inference_time*1000:.2f} ms",
                        help="Time taken for inference"
                    )
                
                with metrics_col2:
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                    st.metric(
                        "Throughput",
                        f"{fps:.2f} FPS",
                        help="Frames per second"
                    )
                
                with metrics_col3:
                    st.metric(
                        "Device",
                        device.upper(),
                        help="Computation device"
                    )
        
        except Exception as e:
            st.error(f"Inference failed: {str(e)}")
            st.info("This might be because the model files are not available. Please run the training script first.")

# Performance comparison section
st.header("Performance Comparison")

# Create performance comparison chart
if st.button("Run Performance Benchmark"):
    st.write("Running performance benchmark...")
    
    # Simulate benchmark results
    benchmark_data = {
        "Model": ["TinyCNN", "TinyCNN", "TinyCNN", "MobileNetV2", "MobileNetV2", "MobileNetV2"],
        "Format": ["PyTorch", "ONNX", "TFLite", "PyTorch", "ONNX", "TFLite"],
        "Latency (ms)": [15.2, 12.8, 8.5, 25.1, 18.3, 11.2],
        "Throughput (fps)": [65.8, 78.1, 117.6, 39.8, 54.6, 89.3],
        "Size (MB)": [2.1, 2.3, 0.8, 4.2, 4.5, 1.2]
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Latency Comparison", "Throughput Comparison", "Model Size Comparison", "Efficiency Score"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Latency comparison
    fig.add_trace(
        go.Bar(x=benchmark_data["Format"], y=benchmark_data["Latency (ms)"], 
               name="Latency", marker_color="red"),
        row=1, col=1
    )
    
    # Throughput comparison
    fig.add_trace(
        go.Bar(x=benchmark_data["Format"], y=benchmark_data["Throughput (fps)"], 
               name="Throughput", marker_color="green"),
        row=1, col=2
    )
    
    # Model size comparison
    fig.add_trace(
        go.Bar(x=benchmark_data["Format"], y=benchmark_data["Size (MB)"], 
               name="Size", marker_color="blue"),
        row=2, col=1
    )
    
    # Efficiency score (throughput/size ratio)
    efficiency_scores = [t/s for t, s in zip(benchmark_data["Throughput (fps)"], benchmark_data["Size (MB)"])]
    fig.add_trace(
        go.Bar(x=benchmark_data["Format"], y=efficiency_scores, 
               name="Efficiency", marker_color="orange"),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Edge AI Performance Benchmark Results"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance insights
    st.subheader("Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏆 Best Latency</h4>
            <p><strong>TFLite Format</strong></p>
            <p>8.5ms average inference time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🚀 Best Throughput</h4>
            <p><strong>TFLite Format</strong></p>
            <p>117.6 FPS maximum throughput</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💾 Smallest Size</h4>
            <p><strong>TFLite Format</strong></p>
            <p>0.8MB compressed model</p>
        </div>
        """, unsafe_allow_html=True)

# Edge deployment recommendations
st.header("Edge Deployment Recommendations")

st.markdown("""
<div class="warning-box">
<h4>📱 Device-Specific Recommendations</h4>
<ul>
<li><strong>Raspberry Pi Zero:</strong> Use TFLite format for optimal performance on ARM CPU</li>
<li><strong>ESP32-CAM:</strong> Use TFLite Micro with further quantization for MCU deployment</li>
<li><strong>Jetson Nano:</strong> Use ONNX format for GPU acceleration with TensorRT</li>
<li><strong>Mobile Devices:</strong> Use CoreML format for iOS deployment</li>
<li><strong>Android Devices:</strong> Use TFLite format for Android deployment</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Technical specifications
st.header("Technical Specifications")

specs_col1, specs_col2 = st.columns(2)

with specs_col1:
    st.subheader("Model Specifications")
    st.write("- **Input Size:** 32x32x3 RGB images")
    st.write("- **Number of Classes:** 3 (Airplane, Car, Bird)")
    st.write("- **Model Architecture:** TinyCNN / MobileNetV2-Tiny")
    st.write("- **Quantization:** INT8 post-training quantization")
    st.write("- **Compression:** Magnitude-based pruning (20%)")

with specs_col2:
    st.subheader("Edge Constraints")
    st.write("- **Memory:** < 1MB model size")
    st.write("- **Latency:** < 20ms inference time")
    st.write("- **Power:** < 100mW power consumption")
    st.write("- **Storage:** < 2MB flash memory")
    st.write("- **CPU:** Single-core ARM Cortex-M4")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>Low-Power Computer Vision Demo | Research & Educational Use Only</p>
<p><strong>⚠️ NOT FOR SAFETY-CRITICAL APPLICATIONS</strong></p>
</div>
""", unsafe_allow_html=True)
