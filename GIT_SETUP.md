# Git Setup Guide

Quick guide to initialize this repository and push to GitHub.

## Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - YOLO Zone Detection v1.0.0"
```

## Push to GitHub

### Option 1: Create New Repository on GitHub

1. Go to https://github.com/new
2. Create a new repository (e.g., `yolo-zone-detection`)
3. **Do NOT** initialize with README, .gitignore, or license
4. Copy the repository URL

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/yolo-zone-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 2: Push to Existing Repository

```bash
# Add remote
git remote add origin <your-repo-url>

# Push
git branch -M main
git push -u origin main
```

## Verify .gitignore

The `.gitignore` file is configured to exclude:
- ✅ Virtual environment (`venv/`)
- ✅ Python cache (`__pycache__/`, `*.pyc`)
- ✅ Model files (`*.pt`, `*_openvino_model/`)
- ✅ IDE files (`.vscode/`, `.idea/`)
- ✅ OS files (`.DS_Store`, `Thumbs.db`)

## What Gets Committed

**Included:**
- All source code (`src/`, `scripts/`, `tools/`)
- Documentation (`docs/`, `README.md`, `CHANGELOG.md`)
- Configuration (`requirements.txt`, `.gitignore`)
- MQTT broker setup (`mqtt-broker/docker-compose.yml`)
- Quick start scripts (`run.bat`, `run.sh`)

**Excluded (will be generated on setup):**
- Model files (`*.pt`, `*_openvino_model/`)
- Virtual environment (`venv/`)
- Cache files (`__pycache__/`)
- User-specific IDE settings

## Repository Size

Expected committed size: **~100 KB** (code only, no models)

Users will download models on first run via:
```bash
python -m scripts.export_models
```

## Recommended GitHub Settings

**Repository Description:**
```
Real-time object detection using YOLOv8 + OpenVINO with zone-based detection and MQTT events
```

**Topics/Tags:**
```
yolov8, openvino, object-detection, mqtt, computer-vision, 
opencv, real-time, zone-detection, intel-cpu, python
```

**README Preview:**
- ✅ Badges for Python version, dependencies
- ✅ Quick start section
- ✅ Feature highlights
- ✅ Installation instructions

## Optional: Add License

```bash
# Add MIT License
curl -o LICENSE https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt

# Edit LICENSE file with your name and year
# Then commit
git add LICENSE
git commit -m "Add MIT License"
```

## Optional: GitHub Actions (CI/CD)

Create `.github/workflows/test.yml` for automated testing:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python -c "import src; print('Imports OK')"
```

## Clone and Use

After pushing, anyone can use:

```bash
git clone https://github.com/YOUR_USERNAME/yolo-zone-detection.git
cd yolo-zone-detection
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m scripts.export_models
python -m src.main --camera 1 --mode balanced
```

---

**Ready to share with the world!** 🚀
