#!/usr/bin/env python3
"""
Setup script for Kannada Fine-tuning Environment
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet.")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["data", "models", "logs", "outputs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "pandas",
        "sklearn",
        "numpy"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✓ All packages imported successfully!")
        return True

def main():
    """Main setup function"""
    print("Kannada Fine-tuning Setup")
    print("=" * 30)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return
    
    # Test imports
    if not test_imports():
        print("Some packages failed to import. Please check your installation.")
        return
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 30)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download dataset: python download_dataset.py")
    print("2. Run training: python train_mbert_kannada.py --data_path data/dataset.csv --task sentiment")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
