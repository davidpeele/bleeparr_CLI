#!/bin/bash

echo "🚀 Starting Bleeparr installation..."

# Update package lists
sudo apt update

# Install system dependencies
echo "📦 Installing FFmpeg..."
sudo apt install -y ffmpeg

# Install Python dependencies
echo "🐍 Installing Python libraries..."
pip3 install -r requirements.txt

echo "✅ Installation complete! You can now run Bleeparr."
