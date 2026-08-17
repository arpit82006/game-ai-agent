# Ball Sort AI

A modular, computer-vision-based automation project for the Ball Sort Puzzle mobile game.

> **Status: Early Development** — Vision pipeline is complete. Solver and automation are not yet implemented.

---

## Current Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11.9 |
| Computer Vision | OpenCV 5.0.0 |
| Android Emulator | BlueStacks 5 (Pie 64-bit) |
| ADB | HD-Adb.exe (BlueStacks bundled) |
| Numerical | NumPy |
| GUI Automation | PyAutoGUI |
| Object Modelling | Python dataclasses |

---

## Current Capabilities

- [x] ADB connection to BlueStacks via `127.0.0.1:5555`
- [x] Live screenshot capture from the Android emulator
- [x] Screenshot loading and preprocessing with OpenCV
- [x] Tube detection using contour analysis
- [x] Tube data model (`Tube` dataclass with geometry and ball list)
- [x] Ball slot detection (position inference from tube geometry)
- [x] Empty/occupied slot detection using HSV thresholding
- [x] Ball color detection using HSV k-means clustering
- [ ] Formal board representation (in progress)
- [ ] Puzzle solver algorithm (not started)
- [ ] Automated game interaction / tap control (not started)

---

## Project Structure

```
ball sorter/
│
├── core/
│   └── adb.py              # ADB wrapper — screenshot capture from emulator
│
├── models/
│   ├── __init__.py
│   └── tube.py             # Tube dataclass model
│
├── vision/
│   ├── __init__.py
│   ├── capture.py          # Window-based screenshot via PyAutoGUI
│   ├── contours.py         # Raw contour visualization
│   ├── detect_balls.py     # Hough-circle ball detection
│   ├── detect_ball_slots.py # Slot position detection and occupancy check
│   ├── detect_colors.py    # Per-slot HSV color detection (k-means)
│   ├── detect_tubes.py     # Tube contour detection and filtering
│   ├── edges.py            # Canny edge visualization
│   ├── measure_contours.py # Contour measurement and display
│   ├── preprocess.py       # Grayscale and blur preprocessing
│   └── read_image.py       # Screenshot loader and image info
│
├── automation/             # (placeholder — not yet implemented)
├── emulator/               # (placeholder — not yet implemented)
├── solver/                 # (placeholder — not yet implemented)
├── tests/                  # (placeholder — not yet implemented)
├── utils/                  # (placeholder — not yet implemented)
├── assets/                 # (placeholder — for future reference images)
│
├── test.py                 # Environment smoke test
├── requirements.txt        # Pinned Python dependencies
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.11.9
- BlueStacks 5 installed (Pie 64-bit instance)
- Ball Sort Puzzle game installed inside BlueStacks

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/game-ai-agent.git
cd game-ai-agent
```

### 2. Create and activate the virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Smoke test — verify the environment

```bash
python test.py
```

Expected output:
```
==================================================
Ball Sort AI Setup Test
==================================================
OpenCV Version : 5.0.0.93
NumPy Version  : 2.x.x
Pillow         : OK
PyAutoGUI      : OK

Everything is working correctly!
```

### Capture a screenshot via ADB

```bash
python -m core.adb
```

Requires BlueStacks running and ADB connected on `127.0.0.1:5555`.

### Run vision modules

```bash
# Detect tubes in a captured screenshot
python -m vision.detect_tubes

# Detect ball slots and occupancy
python -m vision.detect_ball_slots

# Detect ball colors per slot
python -m vision.detect_colors

# View raw contours
python -m vision.contours

# View Canny edges
python -m vision.edges
```

All vision modules read from `screenshots/screen.png` (populated by the ADB capture step).

---

## BlueStacks / ADB Requirement

- **Emulator:** BlueStacks 5, Pie 64-bit
- **ADB path:** `C:\Program Files\BlueStacks_nxt\HD-Adb.exe`
- **Device address:** `127.0.0.1:5555`

To verify ADB connectivity manually:

```bash
"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" devices
```

Expected output should list `127.0.0.1:5555 device`.

---

## Current Limitations

- The solver is **not implemented** — the project cannot play the game automatically yet.
- Color detection is tuned for one specific test level; generalisation across levels is not yet verified.
- There is no game state representation (board model) yet.
- No automated tap/interaction with the game is implemented.
- BlueStacks and the ADB path are hardcoded; no configuration file exists yet.

---

## Future Roadmap

1. **Board representation** — Convert detected tubes and colours into a formal game state.
2. **Solver algorithm** — Implement a BFS/DFS/A* solver for the Ball Sort puzzle.
3. **Move execution** — Send tap commands to BlueStacks via ADB to play moves.
4. **Full automation loop** — Capture → Detect → Solve → Execute → Repeat.
5. **Level generalisation** — Make colour detection robust across all game levels.
6. **Configuration system** — Replace hardcoded paths with a config file.
7. **Tests** — Add unit tests for the vision and solver components.
