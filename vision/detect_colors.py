import cv2
import numpy as np

from vision.detect_tubes import detect_tubes

IMAGE_PATH = "screenshots/screen.png"


def get_ball_color(sample):

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]

    return dominant.astype(int)


def main():

    image = cv2.imread(IMAGE_PATH)

    output = image.copy()

    tubes = detect_tubes(image)

    for tube in tubes:

        top = tube.y + int(tube.height * 0.12)
        bottom = tube.y + int(tube.height * 0.92)

        usable_height = bottom - top

        ball_diameter = tube.width * 0.82

        slots = max(4, round(usable_height / ball_diameter))

        slot_height = usable_height / slots

        for i in range(slots):

            center_x = tube.center[0]
            center_y = int(top + (i + 0.5) * slot_height)

            radius = int(ball_diameter * 0.30)

            sample = image[
                center_y - radius:center_y + radius,
                center_x - radius:center_x + radius
            ]

            if sample.size == 0:
                continue

            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

            s = hsv[:, :, 1].mean()
            v = hsv[:, :, 2].mean()

            has_ball = s > 150 and v > 150

            if has_ball:

                h, s, v = get_ball_color(sample)

                print(
                    f"Tube {tube.id} "
                    f"Slot {i+1} "
                    f"H={h:.0f} "
                    f"S={s:.0f} "
                    f"V={v:.0f}"
                )

                cv2.circle(
                    output,
                    (center_x, center_y),
                    radius,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    str(int(h)),
                    (center_x - 15, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

    cv2.imshow("Detected Colors", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()