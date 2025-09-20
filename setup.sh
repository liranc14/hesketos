#!/bin/bash
set -e

# ----------------------------
# Update system
# ----------------------------
sudo apt-get update
sudo apt-get upgrade -y

# ----------------------------
# Install system dependencies
# ----------------------------
sudo apt-get install -y \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    cloud-guest-utils \
    wget \
    curl \
    software-properties-common

# ----------------------------
# Install NVIDIA driver + CUDA
# ----------------------------
# Detect recommended driver
RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')
echo "Installing NVIDIA driver: $RECOMMENDED_DRIVER"
sudo apt-get install -y $RECOMMENDED_DRIVER

# Optional: install CUDA toolkit (needed for some PyTorch builds)
sudo apt-get install -y nvidia-cuda-toolkit

# Reboot required for driver
echo "NVIDIA driver installed. Reboot may be required to activate GPU."

# ----------------------------
# Create Python virtual environment
# ----------------------------
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# ----------------------------
# Upgrade pip
# ----------------------------
pip install --upgrade pip

# ----------------------------
# Install Python dependencies
# ----------------------------
pip install -r requirements.txt

echo "Setup complete!"
