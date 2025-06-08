# Installation Guide

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- CUDA-compatible GPU (optional but recommended for large datasets)

### Operating System Support
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS 10.15+
- Windows 10/11

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishna1164/xai-intrusion-detection.git
   cd xai-intrusion-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv xai_ids_env
   source xai_ids_env/bin/activate  # On Windows: xai_ids_env\Scripts\activate
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

### Method 2: Development Installation

For contributors and developers:

1. **Clone with development dependencies**
   ```bash
   git clone https://github.com/krishna1164/xai-intrusion-detection.git
   cd xai-intrusion-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv xai_ids_env
   source xai_ids_env/bin/activate  # On Windows: xai_ids_env\Scripts\activate
   ```

3. **Install with development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

### Method 3: Docker Installation

1. **Build Docker image**
   ```bash
   docker build -t xai-intrusion-detection .
   ```

2. **Run container**
   ```bash
   docker run -it --rm -v $(pwd):/app xai-intrusion-detection
   ```

## Dependencies

### Core Dependencies
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `shap>=0.42.0` - SHAP explanations
- `lime>=0.2.0` - LIME explanations
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `joblib>=1.2.0` - Model serialization
- `pyyaml>=6.0` - Configuration files

### Optional Dependencies
- `tensorflow>=2.10.0` - Deep learning models
- `xgboost>=1.7.0` - Gradient boosting
- `lightgbm>=3.3.0` - Light gradient boosting
- `catboost>=1.1.0` - CatBoost algorithm
- `plotly>=5.10.0` - Interactive visualizations

## Verification

After installation, verify the setup:

```bash
python -c "import src; print('Installation successful!')"
```

Run the test suite:
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'shap'**
```bash
pip install shap --upgrade
```

**2. CUDA-related errors**
- Ensure CUDA toolkit is installed
- Check GPU compatibility
- Install appropriate PyTorch/TensorFlow versions

**3. Memory errors during training**
- Reduce batch size in configuration
- Use data sampling for large datasets
- Enable memory optimization flags

**4. Permission errors on Linux/macOS**
```bash
sudo chown -R $USER:$USER ~/.local/
```

### Platform-Specific Notes

**Windows:**
- Install Microsoft Visual C++ 14.0 or greater
- Use Anaconda for easier dependency management

**macOS:**
- Install Xcode command line tools: `xcode-select --install`
- For M1/M2 Macs, use ARM-compatible packages

**Linux:**
- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev build-essential libssl-dev libffi-dev
  ```

## Environment Variables

Set the following environment variables (optional):

```bash
export XAI_IDS_DATA_PATH="/path/to/data"
export XAI_IDS_MODEL_PATH="/path/to/models"
export XAI_IDS_LOG_LEVEL="INFO"
```

## Next Steps

After successful installation:

1. Review the [Usage Guide](usage_guide.md)
2. Explore the example notebooks in `notebooks/`
3. Check the [API Reference](api_reference.md) for detailed documentation
4. See [Deployment Guide](deployment_guide.md) for production setup

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Search existing GitHub issues
3. Create a new issue with detailed error information
4. Join our community discussions

## Updates

To update to the latest version:

```bash
git pull origin main
pip install -e . --upgrade
```

For Docker users:
```bash
docker pull krishna1164/xai-intrusion-detection:latest
```
