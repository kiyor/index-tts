#!/bin/bash
# IndexTTS API Server Startup Script

# Activate conda environment
source /home/kiyor/miniconda3/etc/profile.d/conda.sh
conda activate index-tts

# Install API dependencies
pip install -r requirements_api.txt

# Create outputs directory
mkdir -p outputs

# Start API server
echo "🚀 Starting IndexTTS API Server..."
echo "📍 API Documentation: http://localhost:7871/docs"
echo "🎵 WebUI (if running): http://localhost:7870"
echo "Press Ctrl+C to stop"

python api_server.py --host 0.0.0.0 --port 7871