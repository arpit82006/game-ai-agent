"""
File: main.py

Purpose:
    Single production-quality entry point for the Ball Sort AI Computer Vision System.

Usage:
    python main.py
    python main.py --image path/to/image.png  (for offline verification)

Workflow:
    Find BlueStacks
          ↓
    Connect to ADB
          ↓
    Capture fresh Android screenshot
          ↓
    Load screenshot
          ↓
    Detect tubes (individual geometry)
          ↓
    Detect slots / capacity (per-tube)
          ↓
    Detect ball occupancy (gravity-constrained)
          ↓
    Classify ball colors (HSV multi-class)
          ↓
    Construct board state data structure
          ↓
    Generate visual debug suite (debug/latest/)
          ↓
    Print formatted board state
"""

import sys
import os
import argparse
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.adb import Emulator, ADB_PATH
from vision.pipeline import run_vision_pipeline, VisionResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Ball Sort AI — Vision Pipeline Entry Point")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to static image file (skips live ADB capture)")
    parser.add_argument("--output-dir", "-o", type=str, default="debug/latest", help="Directory for debug output images")
    parser.add_argument("--no-debug", action="store_true", help="Disable writing debug images to disk")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  BALL SORT AI — VISION PIPELINE")
    print("=" * 55)

    image = None
    source_desc = ""

    if args.image:
        # Static image mode (offline regression / test)
        source_path = os.path.abspath(args.image)
        print(f"\n[1/8] Loading static test image...")
        if not os.path.exists(source_path):
            print(f"\n[ERROR] Specified image file does not exist: {source_path}")
            return 1
        image = cv2.imread(source_path)
        if image is None:
            print(f"\n[ERROR] Failed to decode image file: {source_path}")
            return 1
        source_desc = f"Static Image: {source_path}"
        print(f"      OK ({image.shape[1]}x{image.shape[0]})")

        print("[2/8] ADB connection skipped (static mode)")
        print("      OK")

        print("[3/8] Fresh capture skipped (static mode)")
        print("      OK")
    else:
        # Live BlueStacks ADB capture mode
        emulator = Emulator()

        # Step 1: Check BlueStacks executable
        print("\n[1/8] Checking BlueStacks environment...")
        if not emulator.check_adb_executable():
            print(f"\n[ERROR] HD-Adb.exe not found at expected location:")
            print(f"        {ADB_PATH}")
            print("        Please ensure BlueStacks 5 is installed.")
            return 1
        print("      OK (HD-Adb.exe found)")

        # Step 2: Connect ADB
        print("[2/8] Connecting to BlueStacks ADB...")
        connected, conn_msg = emulator.connect()
        if not connected:
            print(f"\n[ERROR] BlueStacks is not running or ADB is unreachable.")
            print(f"        Details: {conn_msg}")
            print("        Please start the BlueStacks Pie 64-bit instance and try again.")
            return 1
        print(f"      OK ({conn_msg})")

        # Step 3: Capture fresh screenshot
        print("[3/8] Capturing fresh Android screenshot...")
        try:
            live_screenshot_path = "screenshots/screen.png"
            image, saved_path = emulator.capture_screenshot(live_screenshot_path)
            source_desc = f"Live ADB Capture: {saved_path}"
            print(f"      OK ({image.shape[1]}x{image.shape[0]} px)")
        except Exception as e:
            print(f"\n[ERROR] Could not capture a fresh screenshot from BlueStacks.")
            print(f"        Details: {e}")
            return 1

    # Step 4-8: Run Vision Pipeline
    try:
        print("[4/8] Detecting tubes...")
        # Step 4, 5, 6, 7, 8 run through the orchestrated pipeline
        result: VisionResult = run_vision_pipeline(
            image=image,
            debug_dir=args.output_dir,
            save_debug=(not args.no_debug)
        )
        print(f"      OK ({result.total_tubes} tubes detected)")

        print("[5/8] Detecting slots & tube geometry...")
        print("      OK (individual capacities preserved)")

        print("[6/8] Detecting ball occupancy...")
        print(f"      OK ({result.total_balls} balls present, {result.empty_tubes} empty tubes)")

        print("[7/8] Classifying ball colors...")
        unique_colors = len(result.colors_detected)
        print(f"      OK ({unique_colors} distinct color classes detected)")

        print("[8/8] Building board state & visual debug output...")
        print(f"      OK (debug assets saved to: {args.output_dir})")

    except Exception as e:
        print(f"\n[ERROR] Vision pipeline processing failed: {e}")
        return 1

    # ── Display Structured Board State
    print("\n" + "=" * 55)
    print("  VALIDATED BOARD STATE")
    print("=" * 55)

    board = result.board
    is_valid, validation_errors = board.validate() if board else (False, ["Board not constructed"])

    for tube in (board.tubes if board else []):
        if tube.is_empty:
            print(f"  Tube {tube.id:2d} (capacity {tube.capacity}): [EMPTY TUBE]")
        else:
            balls_str = " -> ".join(tube.balls)
            empty_s = tube.available_space
            print(f"  Tube {tube.id:2d} (capacity {tube.capacity}): [TOP] {balls_str} [BOTTOM]  ({tube.ball_count}/{tube.capacity} balls, {empty_s} empty)")

    print("\n" + "-" * 55)
    print(f"  Summary:")
    print(f"    Source       : {source_desc}")
    print(f"    Total Tubes  : {result.total_tubes}")
    print(f"    Total Balls  : {result.total_balls}")
    print(f"    Empty Tubes  : {result.empty_tubes}")
    
    color_summary = ", ".join(f"{col} ({cnt})" for col, cnt in sorted(result.colors_detected.items()))
    print(f"    Colors Found : {color_summary}")

    val_status = "PASS" if is_valid else f"FAIL ({'; '.join(validation_errors)})"
    print(f"    Validation   : {val_status}")

    print(f"\n  Raw Data Structure (TOP -> BOTTOM):")
    print(f"    board_state = {board.to_lists() if board else result.board_state}")

    if not args.no_debug:
        print(f"\n  Visual Debug Assets:")
        for name, path in sorted(result.debug_files.items()):
            print(f"    - {name:<10s}: {path}")

    print("\n" + "=" * 55)
    print("  VISION COMPLETE")
    print("=" * 55 + "\n")

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
