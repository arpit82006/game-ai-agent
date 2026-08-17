"""
File: vision/contours.py

Purpose:
    Visualize every contour detected in the screenshot.

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

    print(f"Contours Found : {len(contours)}")

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Contours", output)
    cv2.imshow("Edges", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()