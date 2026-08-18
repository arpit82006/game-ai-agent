"""
File: vision/read_image.py

Purpose:
    Loads the latest screenshot from the emulator
    and displays basic information about it.

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

    height, width, channels = image.shape

    print("=" * 50)
    print("IMAGE INFORMATION")
    print("=" * 50)

    print(f"Width    : {width}")
    print(f"Height   : {height}")
    print(f"Channels : {channels}")

    cv2.imshow("Ball Sort Screenshot", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
