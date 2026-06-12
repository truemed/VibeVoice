#!/bin/bash

source .venv/bin/activate
python simple_tts_api.py --model_path vibevoice/VibeVoice-1.5B --host 0.0.0.0 --port 8000 --device cuda
