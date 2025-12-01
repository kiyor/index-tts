#!/bin/bash

# Activate conda environment
source /home/kiyor/miniconda3/bin/activate index-tts

# Change to the project directory
cd /home/kiyor/index-tts

# Start the webui service
# You can modify the host and port as needed
python webui.py --host 0.0.0.0 --port 7870
