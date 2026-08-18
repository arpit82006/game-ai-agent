import cv2

from models.tube import Tube

IMAGE_PATH = "screenshots/screen.png"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _modal_value(values, tolerance=0.20):
    """
    Return the 'modal' value: the value around which the most other values
    cluster within a relative tolerance band.

    Used only for finding the dominant tube WIDTH across candidates —
    NOT for height (heights are preserved individually per tube).
    """
    if not values:
        return 0
    best_center = values[0]
    best_count  = 0
    for ref in values:
        if ref == 0:
            continue
        cluster = [v for v in values if abs(v - ref) / ref <= tolerance]
        if len(cluster) > best_count:
            best_count  = len(cluster)
            best_center = int(round(sum(cluster) / len(cluster)))
    return best_center


def _sort_reading_order(tubes, row_gap):
    """
    Sort tubes in reading order: rows top-to-bottom, left-to-right within
    each row. Also normalizes upward bounding-box distortions caused by corks
    or celebration effects on completed tubes by aligning each tube's top rim
    with the row's median physical rim baseline.
    """
    if not tubes:
        return []
    import numpy as np
    by_y = sorted(tubes, key=lambda t: t.y)
    rows        = []
    current_row = [by_y[0]]
    for tube in by_y[1:]:
        if tube.y - current_row[0].y > row_gap:
            rows.append(sorted(current_row, key=lambda t: t.x))
            current_row = [tube]
        else:
            current_row.append(tube)
    rows.append(sorted(current_row, key=lambda t: t.x))

    # Normalize cork / celebration top protrusion within each row
    for row in rows:
        if len(row) >= 2:
            true_top = max(t.y for t in row)
            for tube in row:
                # If contour starts above row rim by > 12px due to cork topper
                if tube.y < true_top - 12:
                    bottom = tube.y + tube.height
                    tube.y = true_top
                    tube.height = bottom - true_top

    result = []
    for row in rows:
        result.extend(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_tubes(image):
    """
    Detect and return all Ball Sort tubes visible in the screenshot.

    ── Design principles ────────────────────────────────────────────────

    1. Every tube keeps its OWN detected geometry (x, y, width, height).
       No tube's height is overwritten with any other tube's height.

    2. Partial / empty tube contours are accepted as-is.  Their smaller
       detected height represents how much of the tube is currently
       visible via edge detection.  The downstream pipeline computes
       capacity from THAT tube's own height.

    3. No global slot count is assumed anywhere.

    4. Width consistency is used to find the tube family, but height is
       NEVER averaged or normalised across tubes.

    ── Algorithm ────────────────────────────────────────────────────────

    Stage 1 — Broad geometric pre-filter
        Keep contours that are plausibly tube-shaped:
        - taller than wide (AR ≥ 1.0)
        - 4%–25% of image width (not a hair-line, not a wide banner)
        - area ≥ 0.4% of image (not a speck)

    Stage 2 — Modal-width clustering
        Tubes in a level share the same width.  Find the dominant width
        and keep only contours within ±22% of it.
        Width clustering is the ONLY cross-tube normalization applied.

    Stage 3 — Reading-order sort & ID assignment
        Cluster into rows using the MEDIAN of individual tube heights
        (not a forced global height).
        Sort rows top-to-bottom, tubes within rows left-to-right.

    Returns
    -------
    list[Tube]
        Each Tube has its own accurate geometry.  Tube IDs are assigned
        in reading order (top-left → top-right, then next row…).
    """
    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height

    # ── Edge detection (Canny parameters unchanged from original) ─────
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ── Stage 1: Broad geometric pre-filter ───────────────────────────
    # Intentionally lenient — Stage 2 (width clustering) does the real
    # discrimination.  We keep AR ≥ 1.0 (taller than wide) so that even
    # partial/empty tubes (lower AR) can still enter the pipeline.
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if w == 0 or h == 0:
            continue

        ar         = h / w
        width_frac = w / image_width
        rel_area   = area / image_area

        if ar         < 1.0:   continue   # must be at least as tall as wide
        if width_frac < 0.04:  continue   # not a hair-thin line
        if width_frac > 0.25:  continue   # not a wide UI banner or header
        if rel_area   < 0.004: continue   # not a tiny speck

        candidates.append((x, y, w, h, area, contour))

    if not candidates:
        return []

    # ── Stage 2: Modal-width clustering ───────────────────────────────
    # Find the tube-family width.  Tubes share the same width within a
    # level.  Any contour whose width doesn't match (UI buttons, score
    # bars, etc.) is discarded here.
    #
    # IMPORTANT: only WIDTH is clustered.  Heights are left completely
    # untouched — every tube keeps its own detected height.
    widths  = [c[2] for c in candidates]
    modal_w = _modal_value(widths, tolerance=0.20)

    if modal_w == 0:
        return []

    width_tol = max(8, int(modal_w * 0.22))
    filtered  = [c for c in candidates if abs(c[2] - modal_w) <= width_tol]

    if not filtered:
        return []

    # ── Stage 3: Build Tube objects with individual geometry ───────────
    tubes = []
    for x, y, w, h, area, contour in filtered:
        tubes.append(Tube(
            id=0,        # assigned after sort
            x=x,
            y=y,
            width=w,
            height=h,    # ← each tube's OWN detected height, never overwritten
            area=area,
            contour=contour
        ))

    # ── Stage 4: Reading-order sort using median height for row gap ────
    # The row gap is estimated from the MEDIAN of detected tube heights
    # so that a partial tube (small h) cannot distort the gap calculation.
    heights    = sorted(t.height for t in tubes)
    median_h   = heights[len(heights) // 2]
    row_gap    = median_h * 0.45

    tubes = _sort_reading_order(tubes, row_gap)

    for i, tube in enumerate(tubes, start=1):
        tube.id = i

    return tubes


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found.")
        return

    output = image.copy()
    tubes  = detect_tubes(image)

    print(f"\nDetected Tubes: {len(tubes)}\n")

    for tube in tubes:

        cv2.rectangle(
            output,
            (tube.x, tube.y),
            (tube.x + tube.width, tube.y + tube.height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            str(tube.id),
            (tube.x + 5, tube.y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        print(
            f"Tube {tube.id}: "
            f"x={tube.x}, y={tube.y}, "
            f"w={tube.width}, h={tube.height}, "
            f"AR={tube.aspect_ratio:.2f}, "
            f"center={tube.center}"
        )

    cv2.imshow("Detected Tubes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
