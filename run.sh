#!/bin/bash
# Bat Listener — Launch Script
# Starts the Python bat audio translator application.
#
# Usage:
#   bash run.sh
#
# Requirements: numpy, scipy, sounddevice, soundfile, pygame-ce
# Install: pip3 install -r requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set SDL environment variables for headless / Jetson compatibility.
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-alsa}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-x11}

python3 src/main.py