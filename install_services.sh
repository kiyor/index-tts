#!/bin/bash
# Install IndexTTS systemd services

echo "🚀 Installing IndexTTS systemd services..."

# Copy service files to systemd directory
sudo cp index-tts-api.service /etc/systemd/system/
sudo cp index-tts-webui.service /etc/systemd/system/

# Set proper permissions
sudo chmod 644 /etc/systemd/system/index-tts-api.service
sudo chmod 644 /etc/systemd/system/index-tts-webui.service

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable services (auto-start on boot)
sudo systemctl enable index-tts-api.service
sudo systemctl enable index-tts-webui.service

echo "✅ Services installed and enabled!"
echo ""
echo "📋 Available commands:"
echo "  Start API:     sudo systemctl start index-tts-api"
echo "  Start WebUI:   sudo systemctl start index-tts-webui"
echo "  Stop API:      sudo systemctl stop index-tts-api"
echo "  Stop WebUI:    sudo systemctl stop index-tts-webui"
echo "  Check Status:  sudo systemctl status index-tts-api"
echo "  Check Status:  sudo systemctl status index-tts-webui"
echo "  View Logs:     sudo journalctl -u index-tts-api -f"
echo "  View Logs:     sudo journalctl -u index-tts-webui -f"
echo ""
echo "🔧 Starting services now..."

# Start services
sudo systemctl start index-tts-api
sudo systemctl start index-tts-webui

# Wait a moment for services to start
sleep 3

# Check status
echo "📊 Service Status:"
sudo systemctl status index-tts-api --no-pager -l
echo ""
sudo systemctl status index-tts-webui --no-pager -l

echo ""
echo "🌐 Services should be available at:"
echo "  API Server: http://localhost:7871"
echo "  API Docs:   http://localhost:7871/docs"
echo "  WebUI:      http://localhost:7870"
echo ""
echo "✅ Installation complete!"