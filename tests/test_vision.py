"""
Unit tests for OpenCV vision geometry, slot computation, and capacity inference.
"""

import unittest
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tube import Tube
from vision.detect_ball_slots import compute_tube_slots, detect_tube_occupancy
from vision.detect_tubes import detect_tubes
from vision.pipeline import run_vision_pipeline


class TestVisionSlotGeometry(unittest.TestCase):
    def test_aspect_ratio_to_capacity_mapping(self):
        # 2-capacity tube (aspect ~ 1.95)
        tube_2 = Tube(id=1, x=100, y=200, width=80, height=156, area=12480, contour=None)
        slots_2 = compute_tube_slots(tube_2)
        self.assertEqual(len(slots_2), 2)

        # 3-capacity tube (aspect ~ 2.70)
        tube_3 = Tube(id=2, x=100, y=200, width=80, height=216, area=17280, contour=None)
        slots_3 = compute_tube_slots(tube_3)
        self.assertEqual(len(slots_3), 3)

        # 4-capacity tube (aspect ~ 3.45)
        tube_4 = Tube(id=3, x=100, y=200, width=80, height=276, area=22080, contour=None)
        slots_4 = compute_tube_slots(tube_4)
        self.assertEqual(len(slots_4), 4)

        # 5-capacity tube (aspect ~ 4.20)
        tube_5 = Tube(id=4, x=100, y=200, width=72, height=305, area=21960, contour=None)
        slots_5 = compute_tube_slots(tube_5)
        self.assertEqual(len(slots_5), 5)

        # 5-capacity tube with 73px width (aspect ~ 4.18)
        tube_5b = Tube(id=5, x=100, y=200, width=73, height=305, area=22265, contour=None)
        slots_5b = compute_tube_slots(tube_5b)
        self.assertEqual(len(slots_5b), 5, "Tube 2 with width 73 and height 305 must produce 5 slots")

        # 4-capacity short tube with 72px width (aspect ~ 3.49)
        tube_4_short = Tube(id=6, x=100, y=200, width=72, height=251, area=18072, contour=None)
        slots_4_short = compute_tube_slots(tube_4_short)
        self.assertEqual(len(slots_4_short), 4, "Tube 6 with width 72 and height 251 must produce 4 slots")

    def test_slot_spacing_and_order(self):
        tube = Tube(id=1, x=100, y=200, width=72, height=305, area=21960, contour=None)
        slots = compute_tube_slots(tube)
        self.assertEqual(len(slots), 5)
        # Verify ordered strictly top-to-bottom
        for i in range(len(slots) - 1):
            self.assertLess(slots[i], slots[i + 1])


class TestScreenScreenshotRegression(unittest.TestCase):
    def test_screen_image_detection_if_available(self):
        screen_path = "screenshots/screen.png"
        if not os.path.exists(screen_path):
            self.skipTest("screenshots/screen.png not present in working tree")

        img = cv2.imread(screen_path)
        if img is None:
            self.skipTest("screenshots/screen.png cannot be decoded")

        res = run_vision_pipeline(img, save_debug=False)
        self.assertIn(res.total_tubes, (6, 7, 9), "Must detect 6, 7, or 9 tubes")

        if res.total_tubes == 7:
            self.assertIn(res.total_balls, (20, 24, 25), "Must detect 20-25 balls in 7-tube level")
            self.assertEqual(len(res.board.tubes), 7)
        elif res.total_tubes == 9:
            self.assertEqual(res.total_balls, 28, "Must detect 28 balls in 9-tube level")


class TestColorClassification(unittest.TestCase):
    """
    Unit tests for robust color boundary separation between EMERALD_GREEN and LIGHT_BLUE.
    """
    def test_25_ball_level_emerald_green_and_light_blue_separation(self):
        orig_path = "debug/latest/original.png"
        if not os.path.exists(orig_path):
            orig_path = "01_original.png"
        if not os.path.exists(orig_path):
            self.skipTest("original image not present")

        img = cv2.imread(orig_path)
        if img is None:
            self.skipTest("cannot decode original image")

        res = run_vision_pipeline(img, save_debug=False)
        self.assertEqual(res.total_tubes, 7)
        self.assertEqual(res.total_balls, 25)

        # Expected 5 balls per color
        self.assertEqual(res.board.color_counts, {
            'DARK_PURPLE': 5,
            'EMERALD_GREEN': 5,
            'LIGHT_BLUE': 5,
            'PINK': 5,
            'YELLOW': 5
        })

        # Verify all 5 LIGHT_BLUE balls
        t1 = res.board.get_tube(1)
        t3 = res.board.get_tube(3)
        t5 = res.board.get_tube(5)
        self.assertEqual(t1.balls[2], "LIGHT_BLUE", "Tube 1 Slot 3 must be LIGHT_BLUE")
        self.assertEqual(t1.balls[4], "LIGHT_BLUE", "Tube 1 Slot 5 must be LIGHT_BLUE")
        self.assertEqual(t3.balls[1], "LIGHT_BLUE", "Tube 3 Slot 2 must be LIGHT_BLUE")
        self.assertEqual(t5.balls[0], "LIGHT_BLUE", "Tube 5 Slot 1 must be LIGHT_BLUE")
        self.assertEqual(t5.balls[4], "LIGHT_BLUE", "Tube 5 Slot 5 must be LIGHT_BLUE")

        # Verify all 5 EMERALD_GREEN balls
        t2 = res.board.get_tube(2)
        t4 = res.board.get_tube(4)
        self.assertEqual(t1.balls[3], "EMERALD_GREEN", "Tube 1 Slot 4 must be EMERALD_GREEN")
        self.assertEqual(t2.balls[3], "EMERALD_GREEN", "Tube 2 Slot 4 must be EMERALD_GREEN")
        self.assertEqual(t4.balls[0], "EMERALD_GREEN", "Tube 4 Slot 1 must be EMERALD_GREEN")
        self.assertEqual(t4.balls[2], "EMERALD_GREEN", "Tube 4 Slot 3 must be EMERALD_GREEN")
        self.assertEqual(t5.balls[2], "EMERALD_GREEN", "Tube 5 Slot 3 must be EMERALD_GREEN")

    def test_emerald_green_vs_light_blue_matrix(self):
        from vision.detect_colors import classify_color

        # 1. Normal EMERALD_GREEN (H~76, S~250, V~150)
        self.assertEqual(classify_color(76, 250, 150), "EMERALD_GREEN")

        # 2. Bright EMERALD_GREEN (H=81, S=247, V=189)
        self.assertEqual(classify_color(81, 247, 189), "EMERALD_GREEN")

        # 3. Darker EMERALD_GREEN (H=74, S=252, V=113)
        self.assertEqual(classify_color(74, 252, 113), "EMERALD_GREEN")

        # 4. EMERALD_GREEN with highlight (H=84, S=240, V=190)
        self.assertEqual(classify_color(84, 240, 190), "EMERALD_GREEN")

        # 5. Normal Cyan / LIGHT_BLUE (H=80, S=128, V=246)
        self.assertEqual(classify_color(80, 128, 246), "LIGHT_BLUE")

        # 6. Bright Cyan / LIGHT_BLUE (H=81, S=135, V=239)
        self.assertEqual(classify_color(81, 135, 239), "LIGHT_BLUE")

        # 7. Cyan / LIGHT_BLUE (H=82, S=161, V=218)
        self.assertEqual(classify_color(82, 161, 218), "LIGHT_BLUE")

        # 8. Cyan / LIGHT_BLUE (H=81, S=147, V=231)
        self.assertEqual(classify_color(81, 147, 231), "LIGHT_BLUE")

        # 9. Standard Sky-Blue LIGHT_BLUE (H=97, S=140, V=254)
        self.assertEqual(classify_color(97, 140, 254), "LIGHT_BLUE")

    def test_all_other_color_classes_preserved(self):
        from vision.detect_colors import classify_color
        self.assertEqual(classify_color(0, 255, 200), "RED")
        self.assertEqual(classify_color(175, 255, 200), "RED")
        self.assertEqual(classify_color(12, 255, 200), "ORANGE")
        self.assertEqual(classify_color(25, 255, 200), "YELLOW")
        self.assertEqual(classify_color(50, 255, 200), "GREEN")
        self.assertEqual(classify_color(115, 255, 200), "DARK_BLUE")
        self.assertEqual(classify_color(132, 255, 200), "DARK_PURPLE")
        self.assertEqual(classify_color(147, 255, 200), "MAGENTA")
        self.assertEqual(classify_color(162, 255, 200), "PINK")
        self.assertEqual(classify_color(80, 20, 180), "GRAY")


if __name__ == "__main__":
    unittest.main()
