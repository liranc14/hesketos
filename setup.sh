#!/bin/bash
set -e

# ======================
# Update system
# ======================
sudo apt-get update
sudo apt-get upgrade -y

# ======================
# Install system dependencies
# ======================
sudo apt-get install -y \
    build-essential \
    dkms \
    linux-headers-$(uname -r) \
    python3-venv \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    wget \
    ca-certificates \
    cloud-guest-utils

# ======================
# Install NVIDIA driver from CUDA repo (works for g5.xlarge)
# ======================
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list

sudo apt-get update
sudo apt-get -y install cuda-drivers

echo "NVIDIA driver installed. Reboot the instance to finish setup."
echo "Run 'nvidia-smi' after reboot to check if CUDA is available."

# ======================
# Optional: Install CUDA toolkit (if needed)
# ======================
sudo apt-get -y install cuda
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# ======================
# Optional: Create Python venv and install packages
# ======================
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Setup complete. Reboot recommended to enable GPU."
