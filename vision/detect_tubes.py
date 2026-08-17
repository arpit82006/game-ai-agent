import cv2

from models.tube import Tube

IMAGE_PATH = "screenshots/screen.png"


def detect_tubes(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_height, image_width = image.shape[:2]

    tubes = []
    tube_id = 1

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        width_ratio = w / image_width
        height_ratio = h / image_height

        if area < 10000:
            continue

        if height_ratio < 0.18:
            continue

        if width_ratio > 0.20:
            continue

        tubes.append(
            Tube(
                id=tube_id,
                x=x,
                y=y,
                width=w,
                height=h,
                area=area,
                contour=contour
            )
        )

        tube_id += 1

    tubes.sort(key=lambda t: (t.y, t.x))

    for i, tube in enumerate(tubes, start=1):
            tube.id = i

    return tubes


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found.")
        return

    output = image.copy()

    tubes = detect_tubes(image)

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
            f"x={tube.x}, "
            f"y={tube.y}, "
            f"w={tube.width}, "
            f"h={tube.height}, "
            f"center={tube.center}"
        )

    cv2.imshow("Detected Tubes", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()