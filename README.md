# ✏️ Aether — AR Whiteboard

> Draw in the air. No pen. No touch. Just your hand.

Aether is a real-time augmented reality whiteboard that tracks your index finger through a webcam and renders strokes directly over your camera feed. Built with MediaPipe Hands and OpenCV, it runs entirely on-device with no internet connection required.

---

## Demo

| Draw | Change Color | Clear |
|------|-------------|-------|
| Extend index finger | ✌️ Peace gesture | ✊ Fist gesture |

---

## Features

- 🖊️ **Real-time drawing** — smooth stroke rendering with EMA jitter smoothing
- 🎨 **6 neon colors** — cycle through with a gesture
- 📏 **Adjustable brush size** — thumbs up / point to resize
- ↩️ **Undo** — remove last stroke instantly
- 🧹 **Clear canvas** — wipe everything with a fist
- 💾 **Save as PNG** — exports to `saved/` folder with timestamp
- 📷 **Camera toggle** — switch between AR overlay and pure canvas mode
- ⚡ **Gesture controls** — powered by a trained Random Forest classifier (98%+ accuracy)

---

## How It Works

```
Webcam → MediaPipe Hands → Fingertip Position → EMA Smoother → Canvas Renderer
                        ↘ Gesture Classifier → Action Layer
```

**Drawing detection** uses finger joint geometry — index finger extended + middle finger curled = draw mode. No model needed for this, just landmark math.

**Gesture controls** use a Random Forest classifier trained on 63 normalized 3D hand landmark features (21 landmarks × x, y, z), achieving **98.3% test accuracy** across 9 gesture classes.

---

## Gestures

| Gesture | Action |
|---------|--------|
| ☝️ Index finger extended | Draw |
| ✌️ Peace | Cycle colors |
| 👍 Thumbs up | Brush size + |
| ☝️ Point | Brush size − |
| ✊ Fist | Clear canvas |
| 🤘 Rock | Undo last stroke |
| 👌 OK | Save as PNG |
| ✋ Open palm | Toggle webcam feed |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `H` | Toggle landmark overlay |
| `C` | Clear canvas |
| `Z` | Undo |
| `S` | Save canvas |

---

## Installation

**Prerequisites:** Python 3.9+, Apple Silicon Mac (M1/M2/M3)

```bash
# Clone the repo
git clone https://github.com/yourusername/Aether.git
cd Aether

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mediapipe-silicon opencv-python-headless numpy scikit-learn
```

> **Note:** Uses `mediapipe-silicon` for Apple Silicon compatibility. On Intel Mac or Linux, replace with `mediapipe`.

---

## Usage

```bash
python aether.py
```

Make sure `gesture_rf.pkl` (the trained gesture model) is in the same directory.

**To draw:**
1. Hold your hand in front of the camera
2. Extend your index finger with middle finger curled down
3. Move your hand — strokes follow your fingertip in real time
4. Curl your index finger to pause drawing

---

## Project Structure

```
Aether/
├── aether.py          # Main application
├── gesture_rf.pkl     # Trained gesture classifier
├── saved/             # Exported PNG drawings (auto-created)
└── README.md
```

---

## Performance

| Setting | Value |
|---------|-------|
| Resolution | 640 × 480 |
| MediaPipe model | Lite (complexity=0) |
| Tip smoothing | EMA α=0.35 |
| Min movement threshold | 3px |

Runs at ~30fps on an M-series MacBook with no GPU required.

---

## Tech Stack

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) — hand landmark detection
- [OpenCV](https://opencv.org/) — webcam capture and rendering
- [scikit-learn](https://scikit-learn.org/) — Random Forest gesture classifier
- [NumPy](https://numpy.org/) — landmark normalization and canvas ops

---

## Related Project

The gesture classifier was trained as part of **[Handly](https://github.com/yourusername/Handly)** — a real-time gesture controller for macOS system audio and Apple Music.

---

## License

MIT