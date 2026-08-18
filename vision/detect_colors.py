import cv2
import numpy as np
import sys
import os

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.detect_tubes import detect_tubes
from vision.detect_ball_slots import detect_all_tubes_occupancy

IMAGE_PATH = "screenshots/screen.png"


def get_ball_color(sample_bgr):
    """
    Extract the dominant (Hue, Saturation, Value) of a ball from a BGR patch.

    Filters out extreme specular highlight pixels (V > 240, S < 60)
    to focus on the ball's true body pigment, then clusters via K-Means.

    Returns:
        tuple[int, int, int]: (H, S, V) values in OpenCV range (0-179, 0-255, 0-255).
    """
    if sample_bgr is None or sample_bgr.size == 0:
        return (0, 0, 0)

    hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3)).astype(np.float32)

    # Filter out white specular highlight spots to sample the ball body
    non_highlight = ~((pixels[:, 2] > 240) & (pixels[:, 1] < 60))
    if np.sum(non_highlight) > 10:
        filtered_pixels = pixels[non_highlight]
    else:
        filtered_pixels = pixels

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        filtered_pixels,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)].astype(int)

    return tuple(int(x) for x in dominant)


def classify_color(h, s=255, v=255):
    """
    Classify OpenCV HSV color values into discrete color categories.

    OpenCV HSV Ranges:
      - H: 0 - 179
      - S: 0 - 255
      - V: 0 - 255

    Distinctions:
      - RED:           Deep crimson (H >= 170 or H < 8), high saturation (S > 190), lower brightness.
      - ORANGE:        Warm citrus orange (8 <= H < 18).
      - YELLOW:        Bright yellow/gold (18 <= H < 38).
      - GREEN:         Lime/leaf green (38 <= H < 65, centered at H~50).
      - EMERALD_GREEN: Dark emerald/forest green (65 <= H < 80, centered at H~76, high saturation).
      - LIGHT_BLUE:    Light/cyan blue (80 <= H < 105).
      - DARK_BLUE:     Deep royal/navy blue (105 <= H < 125).
      - DARK_PURPLE:   Deep violet/indigo purple (125 <= H < 140).
      - MAGENTA:       Bright radiant magenta/purple (140 <= H < 155).
      - PINK:          Bright pastel pink (155 <= H < 170).

    Returns:
        str: Standard color name ('PINK', 'RED', 'ORANGE', 'YELLOW', 'GREEN', 'EMERALD_GREEN', 'LIGHT_BLUE', 'DARK_BLUE', 'DARK_PURPLE', 'MAGENTA', 'GRAY').
    """
    # Achromatic mystery balls with low saturation are classified as GRAY
    if s <= 35:
        return "GRAY"

    if h >= 170 or h < 8:
        return "RED"
    elif 8 <= h < 18:
        return "ORANGE"
    elif 18 <= h < 38:
        return "YELLOW"
    elif 38 <= h < 65:
        return "GREEN"
    elif 65 <= h < 105:
        # In the green/cyan/light-blue spectrum:
        # EMERALD_GREEN is characterized by high saturation (S >= 220) and rich deep pigment (S > V or V <= 200).
        # LIGHT_BLUE / Cyan has moderate saturation (S < 220) and high brightness (V >= 200, V > S), or pure blue hue (H >= 90).
        if h < 90 and s >= 220 and (s > v or v <= 200):
            return "EMERALD_GREEN"
        else:
            return "LIGHT_BLUE"
    elif 105 <= h < 125:
        return "DARK_BLUE"
    elif 125 <= h < 140:
        return "DARK_PURPLE"
    elif 140 <= h < 155:
        return "MAGENTA"
    elif 155 <= h < 170:
        return "PINK"
    else:
        # Fallback
        if h >= 165 or h < 10:
            return "RED"
        return "UNKNOWN"


def get_ball_color_name(sample_bgr):
    """
    Convenience function: extract dominant color and return classified name.
    """
    h, s, v = get_ball_color(sample_bgr)
    return classify_color(h, s, v)


# Color mapping for drawing debug overlays
COLOR_BGR_MAP = {
    "RED":           (0,   0,   220),
    "PINK":          (203, 105, 255),
    "MAGENTA":       (220, 40,  200),
    "DARK_PURPLE":   (160, 32,  100),
    "ORANGE":        (0,   140, 255),
    "YELLOW":        (0,   220, 220),
    "GREEN":         (0,   220, 0  ),
    "EMERALD_GREEN": (34,  139, 34 ),
    "LIGHT_BLUE":    (235, 180, 50 ),
    "DARK_BLUE":     (180, 50,  0  ),
    "GRAY":          (160, 160, 160),
    "BLACK":         (20,  20,  20 ),
    "EMPTY":         (80,  80,  80 ),
    "UNKNOWN":       (128, 128, 128)
}


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return

    output = image.copy()
    tubes = detect_tubes(image)
    occupancy_results = detect_all_tubes_occupancy(image, tubes)

    print("\n" + "=" * 55)
    print("  BALL COLOR CLASSIFICATION")
    print("=" * 55)

    for tube in tubes:
        res = occupancy_results[tube.id]
        slots = res["slots"]
        balls_present = res["balls_present"]
        tube_w = tube.width
        r_col = int(tube_w * 0.22)
        ball_radius = int(tube_w * 0.38)

        tube_colors = []
        for i, ((cx, cy), has_ball) in enumerate(zip(slots, balls_present)):
            if has_ball:
                sample = image[
                    max(0, cy - r_col):min(image.shape[0], cy + r_col),
                    max(0, cx - r_col):min(image.shape[1], cx + r_col)
                ]
                h, s, v = get_ball_color(sample)
                color_name = classify_color(h, s, v)
                tube_colors.append(color_name)

                bgr = COLOR_BGR_MAP.get(color_name, (255, 255, 255))
                cv2.circle(output, (cx, cy), ball_radius, bgr, -1)
                cv2.circle(output, (cx, cy), ball_radius, (255, 255, 255), 2)
                cv2.putText(
                    output,
                    color_name[:3],
                    (cx - 16, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1
                )
                print(f"  Tube {tube.id} Slot {i+1}: {color_name:<7} [H={h:3d}, S={s:3d}, V={v:3d}]")
            else:
                tube_colors.append("EMPTY")
                cv2.circle(output, (cx, cy), ball_radius, (80, 80, 80), 1)

        print(f"Tube {tube.id:2d}: {tube_colors}")

    cv2.imwrite("debug/08_detected_colors.png", output)
    print("\nSaved visualization: debug/08_detected_colors.png")


if __name__ == "__main__":
    main()
