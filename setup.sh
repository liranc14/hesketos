#!/bin/bash
set -e

# Update system
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    cloud-guest-utils

# Create venv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python deps from requirements.txt (must exist in same dir)
pip install -r requirements.txt
