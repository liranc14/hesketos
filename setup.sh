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
    cloud-guest-utils \
    gnupg

# ======================
# Install NVIDIA driver + CUDA + cuDNN (for g5.xlarge)
# ======================
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list

sudo apt-get update

# Install NVIDIA driver
sudo apt-get -y install cuda-drivers

# Install CUDA toolkit + cuDNN
sudo apt-get -y install cuda-toolkit-12-2 libcudnn8 libcudnn8-dev

# Update environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# ======================
# Optional: Create Python virtual environment
# ======================
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

# Install requirements if present
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# ======================
# Final messages
# ======================
echo "======================================================"
echo "✅ Setup complete."
echo "⚠️  Reboot the instance now to load NVIDIA drivers and enable GPU."
echo "Run 'nvidia-smi' after reboot to verify CUDA availability."
echo "======================================================"
