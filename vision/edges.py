"""
File: vision/edges.py

Purpose:
    Detect edges in the Ball Sort screenshot.

Author:
    Arpit + ChatGPT
"""

import cv2

IMAGE_PATH = "screenshots/screen.png"


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("❌ Could not load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    cv2.imshow("Original", image)
    cv2.imshow("Edges", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
