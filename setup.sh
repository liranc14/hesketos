#!/bin/bash
# setup_ec2_gpu.sh
# Run as root or with sudo

set -e

echo "=== Updating system packages ==="
apt-get update -y
apt-get upgrade -y

echo "=== Installing general dependencies ==="
apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    unzip \
    ffmpeg \
    libsndfile1 \
    python3-venv \
    python3-pip \
    pkg-config

echo "=== Installing NVIDIA drivers and CUDA ==="
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
apt-get update -y
apt-get -y install cuda

echo "=== Verifying NVIDIA GPU ==="
nvidia-smi

echo "=== Creating Python virtual environment ==="
# Create venv in current directory
python3 -m venv venv

echo "=== Activating virtual environment ==="
source venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python packages from requirements.txt ==="
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found in current directory"
fi

echo "=== Setup complete! ==="
echo "Activate the virtual environment with: source venv/bin/activate"
echo "Then run your transcription script. Set DEVICE='cuda' in the script to use GPU."
