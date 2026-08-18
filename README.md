# Ball Sort AI

A modular, computer-vision, board-state modeling, and automated puzzle-solving system for the **Ball Sort Puzzle** mobile game.

> **Status: Early Development — Vision Pipeline, Formal Board-State Layer & Solver Complete**  
> The system perceives the game screen via ADB, extracts individual tube geometries and slot occupancies, classifies discrete ball colors, constructs an OpenCV-decoupled formal Board representation, validates puzzle invariants, and calculates optimal, shortest-path move sequences using Breadth-First Search (BFS).  
> *Automated screen tapping, ADB move execution, and closed-loop gameplay loops are not yet implemented.*

---

## Target Game

- **Game Title**: [Woody Sort: Color Sort Puzzle](https://play.google.com/store/apps/details?id=com.unicostudio.balltubes&hl=en&pli=1)
- **Developer**: Unico Studio
- **Platform**: Android
- **Google Play Store**: https://play.google.com/store/apps/details?id=com.unicostudio.balltubes&hl=en&pli=1

> **Scope Note**: This project is currently developed and validated against the visual styling, wooden background aesthetic, tube contours, and ball rendering of **Woody Sort: Color Sort Puzzle**. While the underlying architecture is modular and designed to be adaptable to other Ball Sort variants in future iterations, the current computer vision pipeline and calibrated color classifiers are specifically tailored for this game.

---

## Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11.9 | Core runtime environment |
| **Computer Vision** | OpenCV 5.0.0 | Classical edge, contour, and HSV color segmentation (*No YOLO/ML*) |
| **Android Emulator** | BlueStacks 5 (Pie 64-bit) | Game runtime environment |
| **ADB Interface** | `HD-Adb.exe` (Bundled) | Direct ADB communication on `127.0.0.1:5555` |
| **Numerical Processing** | NumPy 2.x | Image array manipulation and statistics |
| **Data Modeling** | Python `dataclasses` | Geometric `Tube` models, formal `Board` / `TubeState`, and `Move` dataclasses |
| **Search Algorithm** | Python standard library | Queue-based Breadth-First Search (BFS) with state deduplication and parent pointers |
| **Unit Testing** | Python `unittest` | 40 automated unit tests covering Board invariants, Move rules, and Search logic |

---

## Architectural Separation of Concerns

```text
BlueStacks 5 (Android Pie 64-bit)
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
               ▼  (Board.from_vision_tubes())
       Formal Board Model
               │
               ▼  (board.validate())
     Validated Board State
               │
               ▼  (solver/solver.py)
   BFS Puzzle Solver Algorithm
               │
               ▼  (solver_result.moves)
     Optimal Move Sequence
               │
               ▼  (Future Automation Stages)
 [ ADB Move Execution ] ──► [ Closed-Loop Verification ]
```

| Layer | Responsibility | Status |
|---|---|---|
| **Core / ADB** | Connects to BlueStacks and pulls guaranteed fresh, cache-busting screencaps. | **Complete** |
| **Vision Pipeline** | Answers: *"What tubes, slots, and ball colors are visually present on screen?"* | **Complete** |
| **Board State Model** | Answers: *"What is the formal, coordinates-free puzzle state?"* | **Complete** |
| **Solver** | Answers: *"What sequence of moves solves the puzzle from this board state?"* | **Complete** |
| **Automation** | Answers: *"How are those moves executed as physical screen taps via ADB?"* | *Not Started* |

---

## Current Capabilities

### Completed Subsystems
- [x] **Live ADB Integration**: Automatically connects to BlueStacks 5 via `HD-Adb.exe` (`127.0.0.1:5555`).
- [x] **Fresh Screenshot Capture**: Pulls live, timestamped frames directly from `/sdcard/` to prevent emulator caching.
- [x] **Consolidated Entry Point**: Single command (`python main.py`) orchestrates capture $\rightarrow$ vision $\rightarrow$ board $\rightarrow$ solver.
- [x] **Offline Static Image Mode**: Supports testing static screenshots via `python main.py --image <path>`.
- [x] **Adaptive Tube Detection**: Detects tubes across single-row and multi-row layouts with reading-order sorting.
- [x] **Per-Tube Geometry & Capacity**: Computes vertical slot centers from each tube's own individual height and width.
- [x] **Gravity-Constrained Ball Occupancy**: Detects ball presence using patch standard deviation and Sobel edge energy.
- [x] **Calibrated 10-Class HSV Color Segmentation**: Accurately classifies 10 discrete color classes with safe decision boundaries.
- [x] **Formal Board-State Model**: OpenCV-decoupled `Board` and `TubeState` models with top-to-bottom stack indexing (`balls[0] = TOP`).
- [x] **Puzzle Invariant Validation**: Verifies capacity limits, non-empty color strings, and puzzle consistency (`board.validate()`).
- [x] **Optimal BFS Puzzle Solver**: Finds shortest-path move sequences using Breadth-First Search with state deduplication.
- [x] **Move Generation & Pruning**: Enforces legal Ball Sort rules, empty tube symmetry reduction, and pure stack pruning.
- [x] **In-Memory Solution Replay**: Step-by-step solution verification confirming ball count and color count conservation.
- [x] **Visual ASCII Board Formatter**: Terminal-safe column renderer displaying tube contents before, during, and after moves.
- [x] **Standalone Offline Demo**: Pure Python solver demonstration (`python -m solver.demo`) without BlueStacks or OpenCV.
- [x] **Comprehensive Test Suite**: 40 unit tests passing (19 Board tests + 21 Solver tests).
- [x] **Real-Game Validation**: Solver-generated move sequences were manually executed on live game levels, achieving 100% puzzle completion.

### Incomplete / Future Subsystems
- [ ] **Automated Move Execution**: Automated touch input dispatching via ADB (`adb shell input tap x y`).
- [ ] **Physical Tap Timing & Animation Sync**: Waiting for in-game pouring animations before dispatching subsequent taps.
- [ ] **Closed-Loop Gameplay**: Autonomous loop (`Capture -> Detect -> Solve -> Tap -> Verify -> Repeat`).
- [ ] **Continuous Real-Time Monitoring**: Real-time screen monitoring and post-move verification.
- [ ] **Native Android Application**: Packaging as an on-device Android service.
- [ ] **Broader Special-Level Color Support**: Automatic classification for unseen holiday/event color palettes.

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

> **Note on Special Levels**: The 10 color classes above represent the validated vocabulary across currently tested levels. Special or event levels in Ball Sort may introduce rare shades or brand-new color variants; the board model and classifier are designed to be extensible to new labels without modifying existing boundaries.

---

## Formal Board-State Layer

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
  - `push(color)` / `pop()`: Standard stack operations for move simulation.
- **`Board`**:
  - `tubes`: Ordered list of `TubeState` objects (T1 .. TN).
  - `num_tubes`, `total_balls`, `empty_tubes_count`, `colors`, `color_counts`.
  - `is_solved`: True when all tubes on the board meet the solved condition.
  - `copy()`: Returns an independent deep copy (safe for tree search).
  - `to_lists()`: Serializes to standard Python `list[list[str]]`.
  - `to_state_tuple()`: Hashable tuple of tuples for solver visited-state sets and memoization.
  - `validate()`: Enforces puzzle integrity and returns `(is_valid, error_list)`.

---

## Puzzle Solver Subsystem

Located in [`solver/`](file:///C:/projects/ball%20sorter/solver), the solver subsystem consumes a `Board` object and computes the shortest valid solution path using Breadth-First Search (BFS):

### Legal Move Rules
A move transferring balls from `src` to `dst` is generated if:
1. `src` is not empty and not already fully solved (`not src.is_solved`).
2. `dst` is not full (`not dst.is_full`).
3. `src.id != dst.id`.
4. `dst` is empty OR `dst.top_color == src.top_color`.
5. Transferred quantity: $\text{ball\_count} = \min(\text{src.top\_same\_color\_count}, \text{dst.available\_space}) \ge 1$.

### Search & Pruning Optimizations
- **State Deduplication**: Explored states are indexed via `board.to_state_tuple()` in a fast hash set.
- **Empty Tube Symmetry**: When multiple empty reserve tubes exist, only the first empty tube is evaluated, eliminating duplicate symmetric search branches.
- **Pure Tube Pruning**: Moving a monochromatic stack into an empty tube is skipped as an isomorphic no-op.
- **Parent-Pointer Backtracking**: Tracks transitions via state keys to reconstruct the minimal move sequence upon reaching `board.is_solved == True`.

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
├── solver/
│   ├── __init__.py            # Export Move, SolverResult, solve, visualizer
│   ├── models.py              # Move and SolverResult data structures
│   ├── move_generator.py      # Legal move generation & symmetry pruning
│   ├── search.py              # Core BFS search algorithm & apply_move()
│   ├── solver.py              # Public solve() API & in-memory replay validator
│   ├── visualizer.py          # Terminal-safe vertical column ASCII board renderer
│   └── demo.py                # Standalone offline puzzle solving demo
│
├── tests/
│   ├── .gitkeep
│   ├── test_board.py          # 19 unit tests for Board & TubeState models
│   └── test_solver.py         # 21 unit tests for Move generator, BFS search, and Replay
│
├── automation/                # (Planned: ADB tap execution & closed-loop controller)
├── emulator/                  # (Placeholder for emulator configurations)
├── utils/                     # (Placeholder for generic helper routines)
├── assets/                    # (Placeholder for reference game assets)
├── debug/
│   └── latest/                # Current runtime visual debug suite (9 images)
│
├── screenshots/               # Runtime screenshot cache (git-ignored)
├── main.py                    # Production entry point: Live ADB -> Vision -> Board -> Solver
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

### 2. Run All Automated Unit Tests
Execute the full 40-test suite covering Board invariants, Move generation, BFS search, and Replay:
```bash
python -m unittest discover tests -v
```

### 3. Run Live Vision & Solver Pipeline (Default)
Ensure BlueStacks 5 is running with a level open, then execute:
```bash
python main.py
```

**Pipeline Workflow:**
1. Connects to BlueStacks via ADB (`127.0.0.1:5555`).
2. Captures fresh screenshot.
3. Runs 8-stage OpenCV vision pipeline.
4. Constructs and validates the formal `Board`.
5. Solves the puzzle using BFS.
6. Verifies the solution via in-memory replay.
7. Prints the complete optimal move sequence.

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

=======================================================
  SOLVER PIPELINE
=======================================================

[9/9] Solving puzzle using Breadth-First Search (BFS)...
      STATUS: SOLVED
      Moves: 13
      States Explored: 2,556
      Solve Time: 0.3460s
      Replay Validation: PASS

=======================================================
  SOLUTION MOVE SEQUENCE (13 moves)
=======================================================
  Move  1: Tube  1 -> Tube  6 | EMERALD_GREEN  x2
  Move  2: Tube  3 -> Tube  6 | EMERALD_GREEN  x1
  Move  3: Tube  3 -> Tube  7 | YELLOW         x1
  Move  4: Tube  1 -> Tube  3 | DARK_PURPLE    x1
  Move  5: Tube  2 -> Tube  1 | MAGENTA        x1
  Move  6: Tube  4 -> Tube  6 | EMERALD_GREEN  x1
  Move  7: Tube  4 -> Tube  7 | YELLOW         x1
  Move  8: Tube  3 -> Tube  4 | DARK_PURPLE    x2
  Move  9: Tube  1 -> Tube  3 | MAGENTA        x2
  Move 10: Tube  5 -> Tube  7 | YELLOW         x2
  Move 11: Tube  2 -> Tube  5 | LIGHT_BLUE     x2
  Move 12: Tube  4 -> Tube  2 | DARK_PURPLE    x3
  Move 13: Tube  3 -> Tube  4 | MAGENTA        x3

=======================================================
  SOLVER COMPLETE — READY FOR AUTOMATION
=======================================================
```

### 4. Live Pipeline + Solver Diagnostics & Visual Replay
To see BFS search metrics and step-by-step ASCII tube states after each move:
```bash
python main.py --verbose-solver
```

#### In-Memory ASCII Replay Sample:
```text
============================================================
  MOVE  1 / 13 : Tube 1 -> Tube 6 | EMERALD_GREEN x2
============================================================
   T1     T2     T3     T4     T5     T6     T7  
  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
  │ . │  │MAG│  │EMG│  │EMG│  │YEL│  │ . │  │ . │
  │ . │  │LBL│  │YEL│  │YEL│  │YEL│  │ . │  │ . │
  │DPU│  │LBL│  │DPU│  │DPU│  │LBL│  │EMG│  │ . │
  │MAG│  │DPU│  │MAG│  │MAG│  │LBL│  │EMG│  │ . │
  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘
  (2/4)  (4/4)  (4/4)  (4/4)  (4/4)  (2/4)  (0/4)
```

### 5. Standalone Offline Solver Demo
Run an isolated in-memory solver demonstration without connecting to BlueStacks or OpenCV:
```bash
python -m solver.demo
```

### 6. Offline Static Image Mode
Analyze an existing screenshot file:
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

1. **Automated Unit Tests**:
   - **Total Tests**: **40 / 40 PASS (100.0%)** in 0.077s.
   - **Board Model Tests** (`tests/test_board.py`): 19 tests validating empty/partial/full tubes, push/pop order, invariants, and serialization.
   - **Solver Tests** (`tests/test_solver.py`): 21 tests validating legal move rules, multi-ball transfer, capacity limits, BFS optimality, unsolvable detection, and conservation laws.
2. **Vision Regression Dataset**:
   - Tested across 7 distinct level configurations.
   - Total balls verified: **119 / 119 balls correctly classified (100.0%)**.
   - *Disclaimer*: 100% on the current dataset represents tested levels; future special levels with unseen background themes, lighting shifts, or unrepresented colors may require classifier additions.
3. **Live Game Validation**:
   - Level A tested: 13 moves, 2,556 states explored, solved in 0.346s.
   - Level B tested: 16 moves, 452 states explored, solved in 0.052s.
   - **Manual Execution Confirmation**: Both solver-generated move sequences were manually played on the live emulator, successfully clearing both levels in Woody Sort with 100% accuracy.

---

## Current Limitations

- **Solver Calculates But Does Not Tap**: The system determines the complete move sequence, but does not yet dispatch automated tap commands to BlueStacks.
- **No Closed-Loop Automation**: The autonomous cycle (`Capture -> Detect -> Solve -> Tap -> Verify -> Repeat`) is not yet assembled.
- **Desktop/Emulator Bound**: Runs as a Python development process communicating with BlueStacks over ADB on Windows; not a standalone mobile APK.
- **Finite Dataset Scope**: Validated on tested game levels; novel event themes or unseen ball colors may require classifier calibration.
- **Default ADB Path**: Defaults to `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` and `127.0.0.1:5555` (configurable in `core/adb.py`).

---

## Future Roadmap

- [x] **Phase 1A: Emulator & ADB Integration** — BlueStacks 5 connection, fresh screenshot capture.
- [x] **Phase 1B: Vision Pipeline** — Adaptive tube detection, slot geometry, occupancy, and 10-color HSV classification.
- [x] **Phase 1C: Formal Board-State Layer** — Decoupled `Board` / `TubeState` models with invariant validation.
- [x] **Phase 2: Puzzle Solver Subsystem** — Optimal BFS search, move generator with symmetry pruning, step-by-step replay validator, and ASCII visualizer.
- [ ] **Phase 3: ADB Move Execution** — Translate calculated `Move` objects into screen tap coordinates (`adb shell input tap x y`) with animation timing delays.
- [ ] **Phase 4: Closed-Loop Automation** — Autonomous gameplay controller verifying post-move states, handling level victory dialogs, and progressing through stages.
- [ ] **Phase 5: Level Generalisation** — Dynamic configuration, broader special-level color vocabularies, and cross-platform emulator support.
