import cv2
import numpy as np
import sys
import os

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.detect_tubes import detect_tubes

IMAGE_PATH = "screenshots/screen.png"


def compute_tube_slots(tube):
    """
    Compute vertical slot centers for a single Tube object based strictly
    on that tube's individual geometry (height, width, y-position).

    Tubes stack balls vertically under gravity.
    The aspect ratio (height / width) geometrically determines the tube's
    capacity (number of balls that fit). For standard Ball Sort tubes:
      - Aspect ratio ~ 1.95 -> 2 slots
      - Aspect ratio ~ 2.70 -> 3 slots
      - Aspect ratio ~ 3.45 -> 4 slots
      - Aspect ratio ~ 4.20 -> 5 slots
      - Aspect ratio ~ 4.95 -> 6 slots

    Returns:
        list[int]: Y-coordinates of slot centers ordered from top to bottom
                   (Slot 1 = topmost slot, Slot N = bottom slot).
                   The length of this list represents the tube's capacity.
    """
    aspect = tube.height / tube.width if tube.width > 0 else 3.5
    capacity = max(1, int(round((aspect - 0.45) / 0.75)))

    bottom_y = tube.y + tube.height - (tube.width * 0.48)
    top_y = tube.y + (tube.width * 0.52)

    if capacity == 1:
        return [int(round(bottom_y))]

    step = (bottom_y - top_y) / (capacity - 1)
    slots = [int(round(top_y + k * step)) for k in range(capacity)]
    return sorted(slots)


def is_ball_present(image, cx, cy, tube_width):
    """
    Determine if a ball is present at a specific slot center (cx, cy).

    Distinguishes an opaque 3D ball from transparent glass / wood background:
    - Balls have 3D spherical shading (specular highlight, shading gradient),
      resulting in standard deviation (>= 12.0) and Sobel edge energy (>= 8.5).
    - Empty tube interiors show the uniform wood background through glass.
      Empty patches with minor wood shadow gradients are rejected by checking
      for the wood background hue (H in [5, 15] with V < 200).

    Returns:
        bool: True if a ball is present, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sample_r = max(6, int(tube_width * 0.20))
    h, w = gray.shape[:2]

    y1, y2 = max(0, cy - sample_r), min(h, cy + sample_r + 1)
    x1, x2 = max(0, cx - sample_r), min(w, cx + sample_r + 1)

    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    mask = np.zeros(patch.shape, dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), sample_r, 255, -1)
    pts = patch[mask > 0]
    if len(pts) == 0:
        return False

    gray_std = float(np.std(pts))
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    edge_y = float(np.mean(np.abs(sobely[mask > 0]))) if np.sum(mask > 0) > 0 else 0.0

    if not (gray_std >= 12.0 and edge_y >= 8.5):
        return False

    # Filter out empty wood background patches showing minor shadow/grain gradient
    if len(image.shape) == 3:
        patch_hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        pts_hsv = patch_hsv[mask > 0]
        mean_h = float(np.mean(pts_hsv[:, 0]))
        mean_s = float(np.mean(pts_hsv[:, 1]))
        mean_v = float(np.mean(pts_hsv[:, 2]))
        # Wood background has high saturation (S > 100), wood hue (H in [5, 15]), and darker brightness (V < 200)
        # (Gray mystery balls have very low saturation S <= 35, and real Orange balls have bright V >= 225)
        if mean_s > 100 and 5 <= mean_h <= 15 and mean_v < 200:
            return False

    return True


def detect_tube_occupancy(image, tube):
    """
    Detect slot positions and occupancy status for a single Tube.

    Evaluates slots for ball presence and enforces physical gravity consistency:
    balls rest in a contiguous stack from the bottom of the tube upward.

    Returns:
        tuple: (slots, balls_present)
            slots: list of (cx, cy) center coordinates for each slot (top to bottom)
            balls_present: list of bool indicating presence of a ball in each slot
    """
    slot_y_coords = compute_tube_slots(tube)
    cx = tube.center[0]

    slots = [(cx, cy) for cy in slot_y_coords]
    raw_present = [is_ball_present(image, cx, cy, tube.width) for _, cy in slots]

    # Enforce physical gravity consistency:
    # A settled tube's balls occupy a contiguous stack from the bottom up.
    balls_present = [False] * len(slots)
    for idx in range(len(slots) - 1, -1, -1):
        if raw_present[idx]:
            balls_present[idx] = True
        else:
            break

    return slots, balls_present


def detect_all_tubes_occupancy(image, tubes):
    """
    Detect slot positions and occupancy for all provided tubes.

    Returns:
        dict[int, dict]: Mapping tube_id -> {
            'slots': list[(cx, cy)],
            'capacity': int,
            'balls_present': list[bool],
            'ball_count': int
        }
    """
    results = {}
    for tube in tubes:
        slots, balls_present = detect_tube_occupancy(image, tube)
        results[tube.id] = {
            "slots": slots,
            "capacity": len(slots),
            "balls_present": balls_present,
            "ball_count": sum(balls_present)
        }
    return results


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return

    output = image.copy()
    tubes = detect_tubes(image)

    print("\n" + "=" * 55)
    print(f"  BALL / SLOT OCCUPANCY DETECTION ({len(tubes)} tubes)")
    print("=" * 55)

    occupancy_results = detect_all_tubes_occupancy(image, tubes)

    for tube in tubes:
        data = occupancy_results[tube.id]
        slots = data["slots"]
        balls_present = data["balls_present"]
        ball_count = data["ball_count"]
        capacity = data["capacity"]
        ball_radius = int(tube.width * 0.38)

        print(f"\nTube {tube.id:2d} (x={tube.x}, y={tube.y}, w={tube.width}, h={tube.height}):")
        print(f"  Capacity      : {capacity} slots")
        print(f"  Balls Present : {balls_present}")
        print(f"  Ball Count    : {ball_count} / {capacity}")

        # Draw tube outline
        cv2.rectangle(
            output,
            (tube.x, tube.y),
            (tube.x + tube.width, tube.y + tube.height),
            (255, 255, 255),
            1
        )
        cv2.putText(
            output,
            f"T{tube.id} ({ball_count}/{capacity})",
            (tube.x, max(20, tube.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        for i, ((cx, cy), has_ball) in enumerate(zip(slots, balls_present)):
            colour = (0, 255, 0) if has_ball else (0, 0, 255)
            status_text = "OCC" if has_ball else "EMP"

            cv2.circle(output, (cx, cy), ball_radius, colour, 2)
            cv2.circle(output, (cx, cy), 3, colour, -1)
            cv2.putText(
                output,
                f"S{i+1}:{status_text}",
                (cx - 20, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                colour,
                1
            )

    print("\n" + "=" * 55)
    print("  SUMMARY BOARD REPRESENTATION")
    print("=" * 55)
    for tube in tubes:
        data = occupancy_results[tube.id]
        occ_str = ", ".join("O" if b else "." for b in data["balls_present"])
        print(f"  Tube {tube.id:2d} [{occ_str}]  ({data['ball_count']}/{data['capacity']} balls)")

    cv2.imwrite("debug/06_slots_occupancy.png", output)
    print("\nSaved debug visualization: debug/06_slots_occupancy.png")


if __name__ == "__main__":
    main()