# Tune-IT

A user-friendly web interface built with Streamlit that enables easy fine-tuning of large language models (LLMs) on custom datasets.

## Overview

Tune-IT simplifies the process of adapting open-source language models to specific domains and tasks. This tool provides an intuitive interface for fine-tuning supported models on custom datasets, allowing users without extensive machine learning expertise to customize models for their own applications.

## Features

### Dataset Management

* Support for multiple data formats (CSV, JSONL, TXT)
* Built-in validation and preprocessing tools
* Optional data augmentation capabilities
* Sample dataset templates for common fine-tuning tasks

### Intuitive Hyperparameter Configuration

* Sensible defaults for quick starts
* Advanced options for experienced users
* Interactive tooltips explaining each parameter's purpose
* Presets for common fine-tuning scenarios (classification, generation, etc.)

### Real-time Training Visualization

* Dynamic loss and accuracy curves
* Live training metrics dashboard
* Generated text samples during training
* Resource utilization monitoring

### Flexible Model Export Options

Download fine-tuned models in multiple formats:

* PyTorch (.pt)
* TensorFlow SavedModel
* GGUF for efficient local inference
* Hugging Face compatible format

## Installation

```bash
# Clone the repository
git clone https://github.com/SakshamVasu/Tune-IT.git

# Move into the project directory
cd Tune-IT

# Create a virtual environment
python -m venv venv

# Activate the virtual environment

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## Requirements

* Python 3.8+
* Streamlit 1.10+
* PyTorch 2.0+ or TensorFlow 2.8+
* Transformers 4.25+
* 16GB+ RAM (32GB+ recommended for larger models)
* CUDA-compatible GPU (8GB+ VRAM recommended for efficient training)

## Usage Guide

### 1. Prepare Your Dataset

Supported formats:

* CSV files with text and label columns
* JSONL files with prompt/completion pairs
* TXT files with appropriate delimiters

### 2. Upload and Configure

* Upload your dataset through the interface.
* Select preprocessing options.
* Choose your preferred model.
* Configure memory and compute constraints.

### 3. Set Hyperparameters

Adjust key parameters:

* Learning rate
* Batch size
* Training epochs
* Output sequence length
* Optimization algorithm

### 4. Train and Monitor

* Start training with a single click.
* Monitor real-time metrics.
* Pause or resume long training sessions.
* Enable early stopping based on validation metrics.

### 5. Export Your Model

* Download the fine-tuned model in your preferred format.
* Access integration code snippets.
* Follow deployment instructions.

## Project Structure

```text
Tune-IT/
├── main.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

## Acknowledgments

* Google for the Gemma family of open models
* Hugging Face for the Transformers ecosystem
* Streamlit for the web application framework
* The open-source machine learning community

## Contact

For questions, suggestions, or bug reports, please open an issue in this repository.

**Repository:** https://github.com/SakshamVasu/Tune-IT
