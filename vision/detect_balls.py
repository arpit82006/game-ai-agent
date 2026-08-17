import cv2

from vision.detect_tubes import detect_tubes

IMAGE_PATH = "screenshots/screen.png"


def main():

    image = cv2.imread(IMAGE_PATH)

    output = image.copy()

    tubes = detect_tubes(image)

    for tube in tubes:

        # Search only inside the tube
        roi = image[
            tube.y:tube.y + tube.height,
            tube.x:tube.x + tube.width
        ]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=35,
            param1=100,
            param2=20,
            minRadius=18,
            maxRadius=45
        )

        if circles is None:
            continue

        circles = circles[0]

        for circle in circles:

            x, y, r = circle

            cx = int(x) + tube.x
            cy = int(y) + tube.y

            cv2.circle(
                output,
                (cx, cy),
                int(r),
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (cx, cy),
                2,
                (0, 0, 255),
                3
            )

    cv2.imshow("Ball Detection", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()