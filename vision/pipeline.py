"""
File: vision/pipeline.py

Purpose:
    Clean, consolidated orchestration layer for the Ball Sort AI vision pipeline.
    Connects existing modular vision components:
      - vision.detect_tubes
      - vision.detect_ball_slots
      - vision.detect_colors
      - models.tube

    Produces structured VisionResult data and complete 8-stage visual debug assets.
"""

from dataclasses import dataclass, field
import os
import cv2
import numpy as np

from models.tube import Tube
from vision.detect_tubes import detect_tubes
from vision.detect_ball_slots import detect_tube_occupancy
from vision.detect_colors import get_ball_color, classify_color, COLOR_BGR_MAP

LABEL_ABBR_MAP = {
    "LIGHT_BLUE":    "LBL",
    "DARK_BLUE":     "DBL",
    "DARK_PURPLE":   "DPU",
    "MAGENTA":       "MAG",
    "PINK":          "PIN",
    "RED":           "RED",
    "YELLOW":        "YEL",
    "ORANGE":        "ORA",
    "GREEN":         "GRE",
    "EMERALD_GREEN": "EMG",
    "EMPTY":         "---",
}


@dataclass
class VisionResult:
    """
    Structured outcome of the computer vision analysis on a screenshot.
    """
    image: np.ndarray
    tubes: list[Tube]
    board_state: list[list[str]]
    total_tubes: int = 0
    total_balls: int = 0
    empty_tubes: int = 0
    colors_detected: dict[str, int] = field(default_factory=dict)
    debug_dir: str = ""
    debug_files: dict[str, str] = field(default_factory=dict)


def run_vision_pipeline(image: np.ndarray, debug_dir: str = "debug/latest", save_debug: bool = True) -> VisionResult:
    """
    Execute the full 8-stage Ball Sort vision pipeline on an in-memory BGR image.

    Stages:
      1. Original Image Validation
      2. Edge Detection (Canny)
      3. Contour Extraction
      4. Tube Detection (Individual geometry preserved)
      5. Slot / Capacity Detection (Per-tube geometry)
      6. Ball Presence / Occupancy Detection (Gravity constrained)
      7. Ball Color Classification (HSV multi-class separation)
      8. Board State Construction & Combined Debug Visualization

    Args:
        image (np.ndarray): Decoded BGR image.
        debug_dir (str): Destination folder for visual debug images.
        save_debug (bool): Whether to write debug images to disk.

    Returns:
        VisionResult: Fully populated data structure containing tubes and board state.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is invalid or empty.")

    img_h, img_w = image.shape[:2]

    # ── Stage 1 & 2: Preprocess & Edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ── Stage 3: Contours
    raw_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = image.copy()
    cv2.drawContours(img_contours, raw_contours, -1, (0, 255, 0), 1)

    # ── Stage 4: Tube Detection
    tubes = detect_tubes(image)
    if not tubes:
        raise RuntimeError("No tubes detected in the screenshot.")

    img_tubes = image.copy()
    for tube in tubes:
        cv2.rectangle(img_tubes, (tube.x, tube.y), (tube.x + tube.width, tube.y + tube.height), (0, 255, 0), 2)
        cv2.putText(img_tubes, f"T{tube.id}", (tube.x + 4, tube.y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # ── Stage 5, 6, 7: Slots, Occupancy & Colors
    img_slots = image.copy()
    img_occupancy = image.copy()
    img_colors = image.copy()
    img_combined = image.copy()

    board_state = []
    colors_counter = {}
    total_balls = 0
    empty_tubes_count = 0

    for tube in tubes:
        # Detect slots and occupancy for this tube
        slots, balls_present = detect_tube_occupancy(image, tube)
        tube.slots = slots
        tube.balls_present = balls_present

        tube_w = tube.width
        ball_radius = int(tube_w * 0.38)
        radius_col = int(tube_w * 0.22)

        tube_color_stack = []
        balls_in_this_tube = 0

        for i, ((cx, cy), has_ball) in enumerate(zip(slots, balls_present)):
            color_name = "EMPTY"
            if has_ball:
                sample = image[
                    max(0, cy - radius_col):min(img_h, cy + radius_col),
                    max(0, cx - radius_col):min(img_w, cx + radius_col)
                ]
                h_val, s_val, v_val = get_ball_color(sample)
                color_name = classify_color(h_val, s_val, v_val)
                tube_color_stack.append(color_name)
                colors_counter[color_name] = colors_counter.get(color_name, 0) + 1
                balls_in_this_tube += 1
                total_balls += 1
            else:
                tube_color_stack.append("EMPTY")

            # ── Draw Slot overlay
            slot_color = (0, 255, 0) if has_ball else (0, 0, 255)
            cv2.circle(img_slots, (cx, cy), ball_radius, slot_color, 2)
            cv2.putText(img_slots, str(i + 1), (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, slot_color, 1)

            # ── Draw Occupancy overlay
            occ_label = "OCC" if has_ball else "EMP"
            occ_color = (0, 200, 0) if has_ball else (0, 0, 200)
            cv2.circle(img_occupancy, (cx, cy), ball_radius, occ_color, 2)
            cv2.putText(img_occupancy, occ_label, (cx - 15, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, occ_color, 1)

            # ── Draw Color & Combined overlay
            draw_col = COLOR_BGR_MAP.get(color_name, (128, 128, 128))
            abbr = LABEL_ABBR_MAP.get(color_name, color_name[:3] if color_name != "EMPTY" else "---")

            cv2.circle(img_colors, (cx, cy), ball_radius, draw_col, -1)
            cv2.putText(img_colors, abbr, (cx - 14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            cv2.circle(img_combined, (cx, cy), ball_radius, draw_col, -1)
            cv2.putText(img_combined, f"S{i+1}:{abbr}", (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        tube.balls = tube_color_stack
        active_balls = [c for c in tube_color_stack if c != "EMPTY"]
        board_state.append(active_balls)

        if balls_in_this_tube == 0:
            empty_tubes_count += 1

    # Tube bounding rectangles on combined overlay
    for tube in tubes:
        cv2.rectangle(img_combined, (tube.x, tube.y), (tube.x + tube.width, tube.y + tube.height), (255, 255, 255), 1)
        cv2.putText(img_combined, f"T{tube.id}", (tube.x + 2, tube.y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ── Stage 8: Pipeline Overview Strip
    TARGET_H = 400
    def resize_h(img, target_h=TARGET_H):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))

    stages = [
        ("ORIGINAL",   resize_h(image)),
        ("EDGES",      cv2.cvtColor(resize_h(edges), cv2.COLOR_GRAY2BGR)),
        ("CONTOURS",   resize_h(img_contours)),
        ("TUBES",      resize_h(img_tubes)),
        ("SLOTS",      resize_h(img_slots)),
        ("OCCUPANCY",  resize_h(img_occupancy)),
        ("COLORS",     resize_h(img_colors)),
        ("COMBINED",   resize_h(img_combined)),
    ]

    strip_frames = []
    for label, frame in stages:
        bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        labeled_frame = np.vstack([bar, frame])
        cv2.rectangle(labeled_frame, (0, 0), (labeled_frame.shape[1] - 1, labeled_frame.shape[0] - 1), (80, 80, 80), 1)
        strip_frames.append(labeled_frame)

    pipeline_strip = np.hstack(strip_frames)

    # ── Save Debug Assets
    debug_files = {}
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs("debug", exist_ok=True)

        saves = {
            "original":  (image, "original.png", "01_original.png"),
            "edges":     (edges, "edges.png", "03_edges.png"),
            "contours":  (img_contours, "contours.png", "04_all_contours.png"),
            "tubes":     (img_tubes, "tubes.png", "05_tubes.png"),
            "slots":     (img_slots, "slots.png", "06_slots.png"),
            "occupancy": (img_occupancy, "occupancy.png", "07_occupancy.png"),
            "colors":    (img_colors, "colors.png", "08_colors.png"),
            "combined":  (img_combined, "combined.png", "09_combined.png"),
            "pipeline":  (pipeline_strip, "pipeline.png", "10_pipeline_strip.png"),
        }

        for key, (img_data, filename, legacy_name) in saves.items():
            primary_path = os.path.join(debug_dir, filename)
            cv2.imwrite(primary_path, img_data)
            debug_files[key] = primary_path

            # Also save to debug/ legacy root
            legacy_path = os.path.join("debug", legacy_name)
            cv2.imwrite(legacy_path, img_data)

    return VisionResult(
        image=image,
        tubes=tubes,
        board_state=board_state,
        total_tubes=len(tubes),
        total_balls=total_balls,
        empty_tubes=empty_tubes_count,
        colors_detected=colors_counter,
        debug_dir=debug_dir,
        debug_files=debug_files
    )
