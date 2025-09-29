# Surgical Path Planning (AI-Assisted Demo)

## Interactive Streamlit demo that:
- Detects a tumor region and locks the target to the yellow box center
- Segments bones and arteries heuristically to create obstacles
- Plans a safe path from a user‑selected entry point to the target with A*
- Runs an AI‑assisted simulation with deviation guidance, alarms, auto‑replan
- Exports a simulation video (MP4)
  
*This is a research/demonstration prototype, not for clinical use.

## Features
- Target lock: detects the yellow annotation box and uses its centroid as target
- Obstacle mask: bones + arteries with tunable sensitivities and safety margin
- Path planning: 16‑direction A* with distance‑transform safety cost
- Interactive UI: click entry point, adjust parameters, replan
  
## Simulation:
- Start/Pause/Step controls, FPS and noise
- Guidance and safety tubes with yellow/red alerts
- Auto‑replan from current tip when outside safety tube
- MP4 generation of the path traversal

## Quick start
1) Python 3.11 recommended
2) Create and activate venv
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS/Linux (bash/zsh):
python3 -m venv .venv
source .venv/bin/activate
3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

## How to use
- Option A: Use the example image button
- Option B: Upload your own CT slice (PNG/JPG)
- Click on the image to set the entry point (blue)
- The app locks the target to the yellow box center (red)

## Tune:
- Bone detection sensitivity
- Artery detection sensitivity
- Safety margin (px)
- Click “Plan Optimal Path” to draw the safe path

## Optional:
- “AI‑Assisted Simulation (beta)”: Start/Pause; adjust FPS, noise
Auto‑replan on red breach
- “Generate Simulation Video” to download MP4

## Repo structure
app.py: Streamlit app and UI
requirements.txt: Python dependencies
temp.png: transient file for processing uploads (ignored by Git)
.gitignore: excludes venv, caches, and large artifacts

## Configuration
- Roboflow API: set API key and model ID in app.py if using hosted detection

## Roadmap
- 3D volume planning over DICOM series
- Robust vessel/bone/airway segmentation models
- Real-time instrument tracking (EM/optical/robot)
- Patient registration and mm-scale safety envelopes
- Procedure logging, metrics, and export

## License
Apache-2.0

## Disclaimer
This software is for research and educational purposes only and is not ready for diagnostic or therapeutic use.
