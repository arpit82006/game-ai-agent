"""
Unit tests for the Ball Sort Automation subsystem.
All tests run in-memory with zero physical ADB commands.
"""

import unittest
from unittest.mock import patch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tube import Tube
from models.board import Board, TubeState
from solver.models import Move
from solver.search import apply_move
from automation import (
    AutomationConfig,
    StepResult,
    ExecutionReport,
    get_tube_tap_point,
    find_tube_by_id,
    execute_move,
    run_full_execution,
    compare_boards
)


class DummyEmulator:
    """Mock emulator tracking tap calls without executing ADB subprocesses."""
    def __init__(self):
        self.taps = []

    def run(self, command):
        if len(command) >= 4 and command[0] == "shell" and command[1] == "input" and command[2] == "tap":
            self.taps.append((int(command[3]), int(command[4])))
        return "OK", ""


class TestAutomationGeometry(unittest.TestCase):
    def setUp(self):
        self.tube1 = Tube(
            id=1, x=100, y=200, width=80, height=300,
            area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32)
        )
        self.tube2 = Tube(
            id=2, x=220, y=200, width=80, height=300,
            area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32)
        )
        self.tubes = [self.tube1, self.tube2]

    def test_normal_and_bonus_tap_delay_defaults(self):
        cfg = AutomationConfig()
        self.assertEqual(cfg.tap_delay, 0.35, "Normal level tap_delay must remain exactly 0.35s")
        self.assertEqual(cfg.bonus_tap_delay, 0.50, "Bonus level bonus_tap_delay must be 0.50s")
        self.assertEqual(cfg.bonus_move_settle_delay, 1.20, "Bonus level settle delay must be 1.20s")
        self.assertEqual(cfg.auto_empty_settle_delay, 1.50, "Auto-empty settle delay must be 1.50s")
        self.assertEqual(cfg.move_settle_delay, 2.80, "Normal level settle delay must remain 2.80s")

    def test_get_tube_tap_point(self):
        pt = get_tube_tap_point(self.tube1)
        self.assertEqual(pt, (140, 365))

    def test_find_tube_by_id(self):
        t = find_tube_by_id(self.tubes, 1)
        self.assertEqual(t, self.tube1)
        self.assertIsNone(find_tube_by_id(self.tubes, 99))

    def test_dry_run_move_execution(self):
        move = Move(from_tube=1, to_tube=2, ball_count=1, color="RED")
        config = AutomationConfig(dry_run=True)
        dummy_em = DummyEmulator()

        ok, msg, src_pt, dst_pt = execute_move(move, self.tubes, config=config, emulator=dummy_em)
        self.assertTrue(ok)
        self.assertIn("[DRY-RUN]", msg)
        self.assertEqual(src_pt, (140, 365))
        self.assertEqual(dst_pt, (260, 365))
        self.assertEqual(len(dummy_em.taps), 0)

    def test_mocked_move_execution(self):
        move = Move(from_tube=1, to_tube=2, ball_count=1, color="RED")
        config = AutomationConfig(tap_delay=0.001, move_settle_delay=0.001, dry_run=False)
        dummy_em = DummyEmulator()

        ok, msg, src_pt, dst_pt = execute_move(move, self.tubes, config=config, emulator=dummy_em)
        self.assertTrue(ok)
        self.assertEqual(len(dummy_em.taps), 2)
        self.assertEqual(dummy_em.taps[0], (140, 365))
        self.assertEqual(dummy_em.taps[1], (260, 365))

    def test_7_tube_coordinate_mapping_and_dispatch(self):
        # Exact 7-tube geometry from the 720x1280 screen
        tubes_7 = [
            Tube(id=1, x=55, y=285, width=84, height=290, area=24360, contour=None),
            Tube(id=2, x=223, y=285, width=83, height=290, area=24070, contour=None),
            Tube(id=3, x=390, y=285, width=84, height=289, area=24276, contour=None),
            Tube(id=4, x=557, y=285, width=84, height=290, area=24360, contour=None),
            Tube(id=5, x=139, y=689, width=83, height=290, area=24070, contour=None),
            Tube(id=6, x=306, y=689, width=84, height=290, area=24360, contour=None),
            Tube(id=7, x=474, y=689, width=83, height=290, area=24070, contour=None),
        ]

        expected_tap_points = {
            1: (97, 444),
            2: (264, 444),
            3: (432, 443),
            4: (599, 444),
            5: (180, 848),
            6: (348, 848),
            7: (515, 848),
        }

        for t in tubes_7:
            pt = get_tube_tap_point(t)
            self.assertEqual(pt, expected_tap_points[t.id], f"Tube {t.id} tap coordinate mismatch")

        # Test execute_move for Tube 1 -> Tube 6
        move = Move(from_tube=1, to_tube=6, ball_count=1, color="YELLOW")
        dummy_em = DummyEmulator()
        config = AutomationConfig(tap_delay=0.001, move_settle_delay=0.001, dry_run=False)

        ok, msg, src_pt, dst_pt = execute_move(move, tubes_7, config=config, emulator=dummy_em)
        self.assertTrue(ok)
        self.assertEqual(src_pt, (97, 444))
        self.assertEqual(dst_pt, (348, 848))
        self.assertEqual(dummy_em.taps, [(97, 444), (348, 848)])

    def test_invalid_tube_id_fails_cleanly(self):
        move = Move(from_tube=1, to_tube=99, ball_count=1, color="RED")
        config = AutomationConfig(dry_run=True)
        ok, msg, _, _ = execute_move(move, self.tubes, config=config)
        self.assertFalse(ok)
        self.assertIn("not found in detected geometry", msg)


class TestBoardComparison(unittest.TestCase):
    def test_identical_boards_match(self):
        b1 = Board.from_lists([["RED", "BLUE"], ["GREEN"], []], capacities=3)
        b2 = Board.from_lists([["RED", "BLUE"], ["GREEN"], []], capacities=3)
        is_match, mismatches = compare_boards(b1, b2)
        self.assertTrue(is_match)
        self.assertEqual(mismatches, [])

    def test_color_mismatch_detected(self):
        b1 = Board.from_lists([["RED", "BLUE"], []], capacities=3)
        b2 = Board.from_lists([["RED", "GREEN"], []], capacities=3)
        is_match, mismatches = compare_boards(b1, b2)
        self.assertFalse(is_match)
        self.assertTrue(any("contents mismatch" in m for m in mismatches))

    def test_ball_count_mismatch_detected(self):
        b1 = Board.from_lists([["RED", "BLUE"], []], capacities=3)
        b2 = Board.from_lists([["RED"], []], capacities=3)
        is_match, mismatches = compare_boards(b1, b2)
        self.assertFalse(is_match)
        self.assertTrue(any("Total ball count mismatch" in m for m in mismatches))

    def test_tube_count_mismatch_detected(self):
        b1 = Board.from_lists([["RED"], []], capacities=3)
        b2 = Board.from_lists([["RED"], [], []], capacities=3)
        is_match, mismatches = compare_boards(b1, b2)
        self.assertFalse(is_match)
        self.assertTrue(any("Tube count mismatch" in m for m in mismatches))

    def test_completed_tube_cork_occlusion_reconciliation(self):
        # Expected: Tube 8 is completed monochromatic [GREEN x4]
        # Actual perceived: Tube 8 has [GREEN x3] due to top-slot cork occlusion
        exp = Board.from_lists([
            ["RED", "BLUE", "YELLOW"],
            ["GREEN", "GREEN", "GREEN", "GREEN"],
            []
        ], capacities=4)
        act = Board.from_lists([
            ["RED", "BLUE", "YELLOW"],
            ["GREEN", "GREEN", "GREEN"],
            []
        ], capacities=4)

        is_match, mismatches = compare_boards(exp, act)
        self.assertTrue(is_match, f"Completed tube cork occlusion must be reconciled safely: {mismatches}")
        self.assertEqual(mismatches, [])


class TestFullExecutionFlow(unittest.TestCase):
    def setUp(self):
        self.tube1 = Tube(id=1, x=100, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        self.tube2 = Tube(id=2, x=220, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        self.tubes = [self.tube1, self.tube2]
        self.config = AutomationConfig(tap_delay=0.001, move_settle_delay=0.001, verification_delay=0.0)

    def test_already_solved_board_stops_with_zero_taps(self):
        board = Board.from_lists([["RED", "RED"], []], capacities=2)
        dummy_em = DummyEmulator()
        report = run_full_execution(board, self.tubes, [], config=self.config, emulator=dummy_em)
        self.assertTrue(report.success)
        self.assertEqual(report.moves_executed, 0)
        self.assertEqual(len(dummy_em.taps), 0)

    def test_full_execution_completes_and_clears_level(self):
        # 1 move solves the puzzle: Tube 1 -> Tube 2 | RED x1
        initial_board = Board.from_lists([["RED", "BLUE", "BLUE"], ["RED"]], capacities=3)
        moves = [Move(from_tube=1, to_tube=2, ball_count=1, color="RED")]
        dummy_em = DummyEmulator()

        # Mock verify_final_state to return solved
        with patch("automation.verifier.verify_final_state") as mock_final:
            mock_final.return_value = (True, "SOLVED_BOARD", "Final board verified in solved state.")
            report = run_full_execution(initial_board, self.tubes, moves, config=self.config, emulator=dummy_em)

        self.assertTrue(report.success)
        self.assertEqual(report.moves_executed, 1)
        self.assertEqual(len(dummy_em.taps), 2)  # 1 move = 2 taps

    def test_final_move_completion_screen_detected(self):
        initial_board = Board.from_lists([["RED", "BLUE"], []], capacities=2)
        moves = [Move(from_tube=1, to_tube=2, ball_count=1, color="RED")]
        dummy_em = DummyEmulator()

        with patch("automation.verifier.verify_final_state") as mock_final:
            mock_final.return_value = (True, "COMPLETION_SCREEN", "Level-complete banner detected.")
            report = run_full_execution(initial_board, self.tubes, moves, config=self.config, emulator=dummy_em)

        self.assertTrue(report.success)
        self.assertEqual(report.moves_executed, 1)
        self.assertEqual(len(dummy_em.taps), 2)

    def test_final_move_ad_transition_detected(self):
        initial_board = Board.from_lists([["RED", "BLUE"], []], capacities=2)
        moves = [Move(from_tube=1, to_tube=2, ball_count=1, color="RED")]
        dummy_em = DummyEmulator()

        with patch("automation.verifier.verify_final_state") as mock_final:
            mock_final.return_value = (True, "AD_OR_NON_GAME_SCREEN", "Post-completion ad transition detected.")
            report = run_full_execution(initial_board, self.tubes, moves, config=self.config, emulator=dummy_em)

        self.assertTrue(report.success)
        self.assertEqual(report.moves_executed, 1)
        self.assertEqual(len(dummy_em.taps), 2)

    def test_final_move_uncertain_halts_safely(self):
        initial_board = Board.from_lists([["RED", "BLUE"], []], capacities=2)
        moves = [Move(from_tube=1, to_tube=2, ball_count=1, color="RED")]
        dummy_em = DummyEmulator()

        with patch("automation.verifier.verify_final_state") as mock_final:
            mock_final.return_value = (False, "FINAL_VERIFICATION_UNCERTAIN", "Final move was dispatched, but completion could not be confirmed.")
            report = run_full_execution(initial_board, self.tubes, moves, config=self.config, emulator=dummy_em)

        self.assertFalse(report.success)
        self.assertEqual(report.moves_executed, 1)  # Final move was dispatched
        self.assertEqual(len(dummy_em.taps), 2)     # Zero extra taps
        self.assertIn("completion could not be confirmed", report.abort_reason)

    def test_intermediate_move_mismatch_halts_strictly(self):
        initial_board = Board.from_lists([["RED", "BLUE"], []], capacities=2)
        moves = [
            Move(from_tube=1, to_tube=2, ball_count=1, color="RED"),
            Move(from_tube=1, to_tube=2, ball_count=1, color="BLUE")
        ]
        dummy_em = DummyEmulator()

        # Mock verify_post_move to fail on Move 1
        with patch("automation.verifier.verify_post_move") as mock_verify:
            mock_verify.return_value = (False, "Simulated board mismatch", None, self.tubes)
            report = run_full_execution(initial_board, self.tubes, moves, config=self.config, emulator=dummy_em)

        self.assertFalse(report.success)
        self.assertEqual(report.moves_executed, 0)
        # Only Move 1 was tapped before halting
        self.assertEqual(len(dummy_em.taps), 2)
        self.assertIn("Simulated board mismatch", report.abort_reason)

    def test_stable_tubes_passed_to_verification_and_never_reindexed(self):
        # 7-tube level with 25 balls (capacity 5)
        initial_board = Board.from_lists([
            ["GREEN", "GREEN", "GREEN", "GREEN"],       # Tube 1: 4 green, will receive 5th
            ["GREEN", "ORANGE", "RED", "LIGHT_BLUE"],   # Tube 2
            ["DARK_PURPLE", "LIGHT_BLUE", "RED", "LIGHT_BLUE"],
            ["ORANGE", "ORANGE", "DARK_PURPLE", "ORANGE", "RED"], # Tube 4
            ["LIGHT_BLUE", "LIGHT_BLUE"],
            ["DARK_PURPLE", "DARK_PURPLE", "DARK_PURPLE"],
            []                                          # Tube 7: empty
        ], capacities=5)

        tubes_7 = [
            Tube(id=i, x=50 + i * 90, y=285 if i <= 4 else 689, width=84, height=305, area=25620, contour=None)
            for i in range(1, 8)
        ]

        moves = [
            Move(from_tube=2, to_tube=1, ball_count=1, color="GREEN"),  # Move 1: completes Tube 1 with 5 GREEN
            Move(from_tube=4, to_tube=2, ball_count=1, color="ORANGE")  # Move 2: unrelated move
        ]
        dummy_em = DummyEmulator()

        # Mock verify_post_move to return the expected board each time with stable_tubes
        call_stable_tubes = []
        def mock_verify_call(expected_b, emulator=None, config=None, stable_tubes=None):
            call_stable_tubes.append(stable_tubes)
            return True, "OK", expected_b, stable_tubes

        with patch("automation.verifier.verify_post_move", side_effect=mock_verify_call), \
             patch("automation.verifier.verify_final_state", return_value=(True, "SOLVED_BOARD", "Cleared")):
            report = run_full_execution(initial_board, tubes_7, moves, config=self.config, emulator=dummy_em)

        self.assertTrue(report.success)
        self.assertEqual(report.moves_executed, 2)
        # Verify that stable_tubes was passed to verify_post_move
        self.assertEqual(len(call_stable_tubes), 1)
        self.assertEqual(len(call_stable_tubes[0]), 7, "Must pass all 7 stable tubes to verification")
        self.assertEqual([t.id for t in call_stable_tubes[0]], [1, 2, 3, 4, 5, 6, 7], "Tube IDs must remain 1..7")


if __name__ == "__main__":
    unittest.main()
