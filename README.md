# Ball Sort AI

A modular, computer-vision and board-state automation project for the **Ball Sort Puzzle** mobile game.

> **Status: Early Development — Vision Pipeline & Formal Board-State Layer Complete**  
> The system reliably connects to BlueStacks, captures fresh Android screens via ADB, extracts individual tube geometries and slot occupancies, classifies 10 discrete ball colors, constructs an OpenCV-decoupled formal Board representation, validates puzzle invariants, and generates visual debug artifacts.  
> *Puzzle solver algorithms and automated move execution are not yet implemented.*

---

## Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11.9 | Core runtime environment |
| **Computer Vision** | OpenCV 5.0.0 | Classical edge, contour, and HSV color segmentation (*No YOLO/ML*) |
| **Android Emulator** | BlueStacks 5 (Pie 64-bit) | Game runtime environment |
| **ADB Interface** | `HD-Adb.exe` (Bundled) | Direct ADB communication on `127.0.0.1:5555` |
| **Numerical Processing** | NumPy 2.x | Image array manipulation and statistics |
| **Data Modeling** | Python `dataclasses` | Geometric `Tube` models and formal `Board` / `TubeState` structures |
| **Unit Testing** | Python `unittest` | Formal board invariant and stack validation test suite |

---

## Architectural Separation of Concerns

```text
BlueStacks (Android Pie 64-bit)
               │
               ▼  (core/adb.py)
       Live Screencap via ADB
               │
               ▼  (vision/pipeline.py)
      OpenCV Vision Pipeline
               │
   ┌───────────┴───────────┐
   ▼                       ▼
Tube Geometry & Slots   Ball Occupancy & Color Classification
   │                       │
   └───────────┬───────────┘
               │
               ▼  (models/board.py)
       Formal Board Model
               │
               ▼  (board.validate())
    Validated Board State
               │
               ▼  (Future Stages)
 [ Solver Algorithm ] ──► [ ADB Move Execution ]
```

| Layer | Responsibility | Status |
|---|---|---|
| **Core / ADB** | Connects to BlueStacks and pulls guaranteed fresh, cache-busting screencaps. | **Complete** |
| **Vision Pipeline** | Answers: *"What tubes, slots, and ball colors are visually present on screen?"* | **Complete** |
| **Board State Model** | Answers: *"What is the formal, coordinates-free puzzle state?"* | **Complete** |
| **Solver** | Answers: *"What sequence of moves solves the puzzle from this board state?"* | *Not Started* |
| **Automation** | Answers: *"How are those moves translated into physical screen taps via ADB?"* | *Not Started* |

---

## Current Capabilities

- [x] **Live ADB Integration**: Automatically connects to BlueStacks 5 via `HD-Adb.exe` (`127.0.0.1:5555`).
- [x] **Fresh Screenshot Capture**: Pulls live, timestamped frames directly from `/sdcard/` to prevent emulator caching.
- [x] **Consolidated Single Entry Point**: Full vision and board pipeline runs via a single command (`python main.py`).
- [x] **Offline Static Image Mode**: Supports testing static screenshots via `python main.py --image <path>`.
- [x] **Adaptive Tube Detection**: Detects tubes across single-row and multi-row layouts with reading-order sorting.
- [x] **Per-Tube Geometry & Capacity**: Computes vertical slot centers from each tube's own individual height and width (no forced global capacity).
- [x] **Gravity-Constrained Ball Occupancy**: Detects ball presence using patch standard deviation and Sobel edge energy, enforcing bottom-to-top gravity stacking.
- [x] **Calibrated 10-Class HSV Color Segmentation**: Accurately classifies 10 discrete color classes with safe multi-feature decision boundaries.
- [x] **Formal Board-State Model**: OpenCV-decoupled `Board` and `TubeState` models with top-to-bottom stack indexing (`balls[0] = TOP`).
- [x] **Puzzle Invariant Validation**: Verifies capacity limits, non-empty color strings, and puzzle consistency (`board.validate()`).
- [x] **Unit Testing Suite**: 19 unit tests validating board invariants, stack operations, copying safety, and edge cases.
- [x] **Visual Debug Suite**: Generates complete 8-stage image pipelines and annotated overlays in `debug/latest/`.
- [ ] **Puzzle Solver**: BFS / DFS / A* move calculation (*Next development stage*).
- [ ] **Automated Move Execution**: Automated tap commands sent to BlueStacks via ADB (*Future*).
- [ ] **Closed-Loop Gameplay**: Capture → Detect → Solve → Execute → Verify loop (*Future*).

---

## Color Classification System

The vision pipeline segments colors using normalized OpenCV HSV values ($H \in [0, 179], S \in [0, 255], V \in [0, 255]$) and distinct decision boundaries:

| Color Class | Hue ($H$) Range | Measured Features / Notes |
|---|:---:|---|
| **`RED`** | $H \ge 170$ or $H < 8$ | Deep crimson ($H \approx 177, S \approx 250, V \approx 150$) |
| **`ORANGE`** | $8 \le H < 18$ | Warm citrus orange ($H \approx 10-12, S \approx 235, V \approx 230$) |
| **`YELLOW`** | $18 \le H < 38$ | Sunny yellow/gold ($H \approx 20-24, S \approx 250, V \approx 225$) |
| **`GREEN`** | $38 \le H < 65$ | Lime / Leaf green ($H \approx 49-50, S \approx 120-188, V \approx 165-230$) |
| **`EMERALD_GREEN`** | $65 \le H < 80$ | Dark pine / Emerald green ($H \approx 73-76, S > 250, V \approx 114-129$) |
| **`LIGHT_BLUE`** | $80 \le H < 105$ | Cyan / Sky blue ($H \approx 81-85 \text{ and } 98-100, S \le 175, V \ge 200$) |
| **`DARK_BLUE`** | $105 \le H < 125$ | Deep royal / Navy blue ($H \approx 110-120$) |
| **`DARK_PURPLE`** | $125 \le H < 140$ | Deep violet / Indigo ($H \approx 132-133, S \approx 216, V \approx 205$) |
| **`MAGENTA`** | $140 \le H < 155$ | Radiant magenta ($H \approx 144-146, S \approx 205, V \approx 248$) |
| **`PINK`** | $155 \le H < 170$ | Pastel pink ($H \approx 160-164, S \approx 137, V \approx 252$) |

> **Note on Level Generalisation**: The 10 color classes above represent the validated vocabulary across currently tested levels. Special or event levels in Ball Sort may introduce rare shades or brand-new color variants; the board model and classifier are designed to be extensible to new labels without modifying existing boundaries.

---

## Formal Board-State Model

Located in [`models/board.py`](file:///C:/projects/ball%20sorter/models/board.py), the board model provides an abstract representation of the puzzle state independent of pixel coordinates, screen resolution, or OpenCV:

### Ball Ordering Convention

$$\text{tube.balls}[0] = \mathbf{TOP\ BALL\ (Exposed,\ Movable)}$$
$$\text{tube.balls}[-1] = \mathbf{BOTTOM\ BALL\ (Base\ of\ Tube)}$$
$$\text{tube.balls} = [] = \mathbf{EMPTY\ TUBE}$$

### Key Methods & Properties

- **`TubeState`**:
  - `capacity`: Maximum ball capacity of the tube.
  - `ball_count`: Number of balls currently in the tube.
  - `available_space`: `capacity - ball_count`.
  - `is_empty` / `is_full`: Boolean capacity flags.
  - `is_pure`: True if all balls in the tube share the same color.
  - `is_solved`: True if empty OR full and monochromatic.
  - `top_color`: Returns `balls[0]` or `None`.
  - `top_same_color_count`: Count of contiguous identical balls from the top down.
  - `push(color)` / `pop()`: Standard stack operations for solver move exploration.
- **`Board`**:
  - `tubes`: Ordered list of `TubeState` objects (T1 .. TN).
  - `num_tubes`, `total_balls`, `empty_tubes_count`, `colors`, `color_counts`.
  - `is_solved`: True when all tubes on the board meet the solved condition.
  - `copy()`: Returns an independent deep copy (safe for tree search).
  - `to_lists()`: Serializes to standard Python `list[list[str]]`.
  - `to_state_tuple()`: Hashable tuple of tuples for solver visited-state sets and memoization.
  - `validate()`: Enforces puzzle integrity and returns `(is_valid, error_list)`.

---

## Project Structure

```text
ball sorter/
├── core/
│   └── adb.py                 # BlueStacks detection, ADB connection & fresh screenshot pull
│
├── models/
│   ├── __init__.py            # Export Tube, TubeState, Board
│   ├── board.py               # Solver-ready Board & TubeState models + validation
│   └── tube.py                # OpenCV geometric Tube dataclass
│
├── vision/
│   ├── __init__.py
│   ├── pipeline.py            # Consolidated 8-stage vision orchestrator
│   ├── detect_tubes.py        # Contour filtering, modal width clustering, row sorting
│   ├── detect_ball_slots.py   # Slot geometry inference & Sobel/Std occupancy detection
│   ├── detect_colors.py       # 10-class HSV color segmentation & BGR palette mapping
│   ├── capture.py             # PyAutoGUI window capture utility (development)
│   ├── contours.py            # Standalone contour visualizer
│   ├── detect_balls.py        # Hough circle ball detection (development)
│   ├── edges.py               # Canny edge visualizer
│   ├── measure_contours.py    # Geometric contour metric reporter
│   ├── preprocess.py          # Grayscale & Gaussian blur utility
│   └── read_image.py          # Image dimension inspector
│
├── tests/
│   ├── .gitkeep
│   └── test_board.py          # 19 unit tests for Board & TubeState models
│
├── solver/                    # (Planned: BFS/DFS/A* puzzle solver algorithms)
├── automation/                # (Planned: ADB tap execution & closed-loop controller)
├── emulator/                  # (Placeholder for emulator configurations)
├── utils/                     # (Placeholder for generic helper routines)
├── assets/                    # (Placeholder for reference game assets)
├── debug/
│   ├── latest/                # Current runtime visual debug suite (9 images)
│   └── _audit_pipeline.py     # Standalone pipeline validation script
│
├── screenshots/               # Runtime screenshot cache (git-ignored)
├── main.py                    # Production entry point: Live ADB -> Vision -> Validated Board
├── test.py                    # Environment & dependency verification test
├── requirements.txt           # Pinned Python package dependencies
├── .gitignore                 # Git ignore rules for virtual environments & debug assets
└── README.md                  # Project documentation
```

---

## Setup & Installation

### Prerequisites

- **OS**: Windows 10 / 11
- **Python**: 3.11.9
- **Emulator**: BlueStacks 5 (Pie 64-bit instance running Ball Sort Puzzle)
- **ADB Path**: `C:\Program Files\BlueStacks_nxt\HD-Adb.exe`

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/game-ai-agent.git
cd game-ai-agent
```

### 2. Create and Activate Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```cmd
# Windows Command Prompt
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Environment Smoke Test

Verify that OpenCV, NumPy, and image libraries are functioning:

```bash
python test.py
```

### 2. Run Board Model Unit Tests

Execute the 19 unit tests covering board invariants, stack operations, copying, and validation:

```bash
python -m unittest discover tests -v
```

### 3. Run Live Vision & Board Pipeline (Default)

Ensure BlueStacks 5 is running with a level open, then execute:

```bash
python main.py
```

**What this does automatically:**
1. Verifies BlueStacks `HD-Adb.exe` availability.
2. Connects to the emulator via ADB (`127.0.0.1:5555`).
3. Captures a guaranteed fresh, un-cached screenshot directly from Android.
4. Executes the full 8-stage OpenCV vision pipeline.
5. Constructs the formal `Board` data structure.
6. Validates puzzle invariants (`Validation: PASS`).
7. Generates visual debug PNGs in `debug/latest/`.
8. Prints the formatted board state to the terminal.

#### Example Terminal Output:

```text
=======================================================
  BALL SORT AI — VISION PIPELINE
=======================================================

[1/8] Checking BlueStacks environment...
      OK (HD-Adb.exe found)
[2/8] Connecting to BlueStacks ADB...
      OK (Connected to 127.0.0.1:5555)
[3/8] Capturing fresh Android screenshot...
      OK (720x1280 px)
[4/8] Detecting tubes...
      OK (7 tubes detected)
[5/8] Detecting slots & tube geometry...
      OK (individual capacities preserved)
[6/8] Detecting ball occupancy...
      OK (20 balls present, 2 empty tubes)
[7/8] Classifying ball colors...
      OK (5 distinct color classes detected)
[8/8] Building board state & visual debug output...
      OK (debug assets saved to: debug/latest)

=======================================================
  VALIDATED BOARD STATE
=======================================================
  Tube  1 (capacity 4): [TOP] EMERALD_GREEN -> EMERALD_GREEN -> DARK_PURPLE -> MAGENTA [BOTTOM]  (4/4 balls, 0 empty)
  Tube  2 (capacity 4): [TOP] MAGENTA -> LIGHT_BLUE -> LIGHT_BLUE -> DARK_PURPLE [BOTTOM]  (4/4 balls, 0 empty)
  Tube  3 (capacity 4): [TOP] EMERALD_GREEN -> YELLOW -> DARK_PURPLE -> MAGENTA [BOTTOM]  (4/4 balls, 0 empty)
  Tube  4 (capacity 4): [TOP] EMERALD_GREEN -> YELLOW -> DARK_PURPLE -> MAGENTA [BOTTOM]  (4/4 balls, 0 empty)
  Tube  5 (capacity 4): [TOP] YELLOW -> YELLOW -> LIGHT_BLUE -> LIGHT_BLUE [BOTTOM]  (4/4 balls, 0 empty)
  Tube  6 (capacity 4): [EMPTY TUBE]
  Tube  7 (capacity 4): [EMPTY TUBE]

-------------------------------------------------------
  Summary:
    Source       : Live ADB Capture: screenshots/screen.png
    Total Tubes  : 7
    Total Balls  : 20
    Empty Tubes  : 2
    Colors Found : DARK_PURPLE (4), EMERALD_GREEN (4), LIGHT_BLUE (4), MAGENTA (4), YELLOW (4)
    Validation   : PASS

  Raw Data Structure (TOP -> BOTTOM):
    board_state = [
        ['EMERALD_GREEN', 'EMERALD_GREEN', 'DARK_PURPLE', 'MAGENTA'],
        ['MAGENTA', 'LIGHT_BLUE', 'LIGHT_BLUE', 'DARK_PURPLE'],
        ['EMERALD_GREEN', 'YELLOW', 'DARK_PURPLE', 'MAGENTA'],
        ['EMERALD_GREEN', 'YELLOW', 'DARK_PURPLE', 'MAGENTA'],
        ['YELLOW', 'YELLOW', 'LIGHT_BLUE', 'LIGHT_BLUE'],
        [],
        []
    ]

=======================================================
  VISION COMPLETE
=======================================================
```

### 4. Offline Static Image Mode

To analyze an existing screenshot without connecting to BlueStacks:

```bash
python main.py --image path/to/screenshot.png
```

---

## Visual Debug Output

Every run of `main.py` dynamically populates [`debug/latest/`](file:///C:/projects/ball%20sorter/debug/latest) with visual inspection images:

| File | Description |
|---|---|
| `debug/latest/original.png` | Fresh screenshot captured from Android |
| `debug/latest/edges.png` | Canny edge detection output |
| `debug/latest/contours.png` | Raw external contour extraction |
| `debug/latest/tubes.png` | Filtered tube bounding boxes and assigned tube IDs |
| `debug/latest/slots.png` | Geometric slot centers and slot numbers per tube |
| `debug/latest/occupancy.png` | Occupied (`OCC`) vs. Empty (`EMP`) slot markers |
| `debug/latest/colors.png` | Detected color fills and three-letter abbreviations |
| `debug/latest/combined.png` | Complete annotated game overlay |
| `debug/latest/pipeline.png` | 8-stage side-by-side horizontal pipeline overview strip |

---

## Testing & Validation Summary

1. **Board Model Unit Tests**:
   - 19 automated unit tests in `tests/test_board.py`.
   - **Result**: **19 / 19 PASS (100.0%)**.
2. **Vision Regression Dataset**:
   - Tested across 7 distinct level configurations (varying tube heights, capacities, single-row and two-row layouts, empty and partially filled tubes).
   - Total balls verified: **119 / 119 balls correctly classified (100.0%)**.
   - *Disclaimer*: While 100% accuracy was achieved on the validation dataset, future special levels with unseen background themes, lighting shifts, or unrepresented colors may require incremental classifier calibration.

---

## Current Limitations

- **Solver Not Implemented**: The system accurately detects and represents the game board, but does not yet calculate solutions or move trees.
- **No Move Execution**: No automated screen tapping or ADB input commands are executed during gameplay.
- **No Closed-Loop Loop**: The continuous cycle (`Capture -> Detect -> Solve -> Tap -> Verify -> Repeat`) is not yet assembled.
- **Desktop/Emulator Bound**: Runs as a Python development process communicating with BlueStacks over ADB on Windows; not a standalone mobile APK.
- **Hardcoded Default ADB Path**: Defaults to `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` and `127.0.0.1:5555` (configurable in `core/adb.py`).

---

## Future Roadmap

- [x] BlueStacks 5 + ADB connection and fresh screenshot capture
- [x] Classical OpenCV vision pipeline (tubes, geometry, slots, occupancy, colors)
- [x] 10-class calibrated HSV color segmentation
- [x] Formal solver-ready Board and TubeState data structures
- [x] Board validation invariants and unit test suite
- [x] Consolidated `main.py` entry point and visual debug pipeline
- [ ] **Phase 2: Puzzle Solver** — Implement BFS, DFS, and A* solvers with move pruning and transposition tables.
- [ ] **Phase 3: Move Execution** — Implement ADB touch input dispatching (`adb shell input tap x y`) based on calculated solution moves.
- [ ] **Phase 4: Closed-Loop Automation** — Autonomous loop to solve levels, verify outcomes, handle animations, and advance stages.
- [ ] **Phase 5: Level Generalisation** — Dynamic configuration, broader special-level color vocabularies, and cross-platform emulator support.
