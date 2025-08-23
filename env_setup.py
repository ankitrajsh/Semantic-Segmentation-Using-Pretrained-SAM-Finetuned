#!/usr/bin/env python3
"""
Environment setup script for Prostate Cancer Semantic Segmentation project.

This script sets up the environment, installs dependencies, and prepares
the project for training and evaluation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"  Primary GPU: {gpu_name}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False


def create_directory_structure():
    """Create necessary directories."""
    print("\nCreating directory structure...")
    
    directories = [
        "data/images",
        "data/masks",
        "models", 
        "results",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    return True


def create_sample_config():
    """Create sample configuration file."""
    print("\nCreating sample configuration...")
    
    config_content = """{
  "model": {
    "name": "facebook/sam-vit-base",
    "num_classes": 6,
    "freeze_encoder": false
  },
  "training": {
    "batch_size": 2,
    "num_epochs": 10,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10
  },
  "data": {
    "image_dir": "data/images",
    "mask_dir": "data/masks",
    "patch_size": 256,
    "train_split": 0.8
  },
  "classes": {
    "0": "Background",
    "1": "Benign", 
    "2": "Gleason_3",
    "3": "Gleason_4",
    "4": "Gleason_5", 
    "5": "Stroma"
  }
}"""
    
    with open("config.json", "w") as f:
        f.write(config_content)
    
    print("✓ Created: config.json")
    return True


def create_data_readme():
    """Create README for data directory.""" 
    print("\nCreating data README...")
    
    readme_content = """# Data Directory

This directory should contain your prostate histopathology data.

## Structure

```
data/
├── images/          # Training image patches (256x256 recommended)
└── masks/           # Corresponding mask patches
```

## File Format

- **Images**: .jpg, .png, or .tiff files
- **Masks**: .png files with class labels as pixel values
- **Naming**: Image and mask files should have matching names

## Class Labels

The mask files should contain pixel values corresponding to:

- 0: Background (Black)
- 1: Benign tissue (Blue)
- 2: Gleason pattern 3 (Green)
- 3: Gleason pattern 4 (Yellow)
- 4: Gleason pattern 5 (Red)
- 5: Stroma (Purple)

## Getting Started

1. Place your image patches in `data/images/`
2. Place corresponding mask patches in `data/masks/`
3. Ensure file names match between images and masks
4. Run the training script: `python workflow_runner.py`

## Data Preprocessing

If you have large histopathology images, use the preprocessing scripts:

- `Patches.py` - Extract patches from large images
- `patch_for_mask.py` - Process corresponding masks
- `Resize.py` - Resize images if needed

## Quality Checks

Before training, verify your data:

```python
from src.dataset import analyze_dataset_statistics
analyze_dataset_statistics("data/images", "data/masks")
```
"""
    
    with open("data/README.md", "w") as f:
        f.write(readme_content)
    
    print("✓ Created: data/README.md")
    return True


def verify_installation():
    """Verify that all components are installed correctly."""
    print("\nVerifying installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # Test transformers
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        # Test SAM
        from transformers import SamModel, SamProcessor
        print("✓ SAM model imports work")
        
        # Test MONAI
        import monai
        print(f"✓ MONAI {monai.__version__}")
        
        # Test other key packages
        import cv2
        import matplotlib
        import sklearn
        import pandas
        print("✓ All key packages imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("PROSTATE CANCER SEGMENTATION SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        print("Failed to create directories")
        sys.exit(1)
    
    # Create configuration files
    create_sample_config()
    create_data_readme()
    
    # Check GPU (if packages are installed)
    try:
        check_gpu()
    except:
        print("Note: Install requirements first to check GPU availability")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Place your data in the data/ directory")
    print("3. Run: python workflow_runner.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
