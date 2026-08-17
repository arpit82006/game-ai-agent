import cv2
import numpy as np

from vision.detect_tubes import detect_tubes

IMAGE_PATH = "screenshots/screen.png"


def main():

    image = cv2.imread(IMAGE_PATH)

    output = image.copy()

    tubes = detect_tubes(image)

    for tube in tubes:

        # Ignore tube rim
        top = tube.y + int(tube.height * 0.12)

        # Ignore rounded bottom
        bottom = tube.y + int(tube.height * 0.92)

        usable_height = bottom - top

        ball_diameter = tube.width * 0.82

        slots = max(4, round(usable_height / ball_diameter))

        slot_height = usable_height / slots

        for i in range(slots):

            center_x = tube.center[0]
            center_y = int(top + (i + 0.5) * slot_height)

            radius = int(ball_diameter * 0.42)

            # -----------------------------
            # Create circular mask
            # -----------------------------
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            cv2.circle(
                mask,
                (center_x, center_y),
                radius,
                255,
                -1
            )

            # Mean colour inside slot
            b, g, r, _ = cv2.mean(image, mask=mask)

            hsv = cv2.cvtColor(
                np.uint8([[[b, g, r]]]),
                cv2.COLOR_BGR2HSV
            )[0][0]

            h = hsv[0]
            s = hsv[1]
            v = hsv[2]

            # -----------------------------
            # Simple ball detector
            # -----------------------------
            has_ball = (
                s > 150 and
                v > 150
            )

            colour = (0, 255, 0)

            if not has_ball:
                colour = (0, 0, 255)

            cv2.circle(
                output,
                (center_x, center_y),
                radius,
                colour,
                2
            )

            cv2.putText(
                output,
                str(i + 1),
                (center_x - 8, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1
            )

            print(
                f"Tube {tube.id} "
                f"Slot {i+1} | "
                f"H={h:3d} "
                f"S={s:3d} "
                f"V={v:3d} "
                f"Ball={has_ball}"
            )

    cv2.imshow("Ball Slots", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()