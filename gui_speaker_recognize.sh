#!/usr/bin/env bash
set -euo pipefail

# Determine script directory and change to SpeechEnhance
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/SpeakerRecognition"

# Forward all arguments to Python script
python reference_gui_pyqt.py