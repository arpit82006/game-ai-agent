"""
File: vision/preprocess.py

Purpose:
    Basic image preprocessing for Ball Sort AI.

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()