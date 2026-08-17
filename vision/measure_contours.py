"""
File: vision/measure_contours.py

Purpose:
    Measure every contour found by OpenCV.

Author:
    Arpit + ChatGPT
"""

import cv2

IMAGE_PATH = "screenshots/screen.png"


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found.")
        return

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("\nID   X    Y    W    H    Area")

    print("-" * 45)

    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        print(
            f"{i:02d}  "
            f"{x:3d}  "
            f"{y:3d}  "
            f"{w:3d}  "
            f"{h:3d}  "
            f"{int(area):6d}"
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )

    cv2.imshow("Bounding Boxes", output)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()