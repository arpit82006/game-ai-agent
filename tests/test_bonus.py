"""
File: tests/test_bonus.py
Purpose:
    Comprehensive unit tests for the Special / Bonus Level subsystem:
    - Mode detection
    - Reveal move planning & prioritization
    - Structural verification
    - Mystery count tracking
    - Stalling & abort safety
"""

import unittest
from unittest.mock import patch
import numpy as np
from models.tube import Tube
from models.board import Board, TubeState
from solver.models import Move
from bonus.detector import is_bonus_level
from bonus.models import RevealMove, BonusRevealReport, RevealStepContext, DestinationMode
from bonus.planner import get_legal_reveal_moves, select_best_reveal_move
from bonus.verifier import verify_reveal_transition
from bonus.controller import run_bonus_reveal_loop, run_bonus_reveal_single_step, create_reveal_context
from vision.pipeline import run_bonus_vision
from automation.models import AutomationConfig
from automation.adb_input import tap, get_total_physical_taps, clear_tap_history


class TestBonusDetector(unittest.TestCase):
    def test_normal_board_is_not_bonus_level(self):
        normal_board = Board.from_lists([
            ["RED", "BLUE", "GREEN"],
            ["BLUE", "RED", "GREEN"],
            []
        ], capacities=3)
        self.assertFalse(is_bonus_level(normal_board))

    def test_board_with_gray_is_bonus_level(self):
        bonus_board = Board.from_lists([
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)
        self.assertTrue(is_bonus_level(bonus_board))

    def test_none_board_returns_false(self):
        self.assertFalse(is_bonus_level(None))


class TestBonusPlanner(unittest.TestCase):
    def test_reveal_move_generation_and_prioritization(self):
        # Board with Tube 1 having PINK over mystery, and Tube 2 having matching PINK with available space
        board = Board.from_lists([
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK"],  # Partial tube with 3 available spaces
            [],
            []
        ], capacities=4)

        moves = get_legal_reveal_moves(board)
        self.assertGreater(len(moves), 0)

        best_move = select_best_reveal_move(board)
        self.assertIsNotNone(best_move)

        # The best move should be merging Tube 4 (PINK) onto Tube 5 (PINK)
        # because merging matching colors has higher priority (20) than moving to an empty tube (10)
        self.assertEqual(best_move.move.color, "PINK")
        self.assertEqual(best_move.move.from_tube, 4)
        self.assertEqual(best_move.move.to_tube, 5)
        self.assertTrue(best_move.will_expose_mystery)
        self.assertEqual(best_move.priority, 20)

    def test_cannot_move_gray_directly(self):
        # A tube whose top ball is GRAY should NEVER generate outgoing moves
        board = Board.from_lists([
            ["GRAY", "GRAY"],
            ["RED", "GRAY"],
            []
        ], capacities=2)

        moves = get_legal_reveal_moves(board)
        # Tube 1 (top ball GRAY) should not be the source of any move
        for m in moves:
            self.assertNotEqual(m.move.from_tube, 1)

    def test_cannot_pour_onto_gray(self):
        # Cannot pour a colored ball onto a tube whose top ball is GRAY
        board = Board.from_lists([
            ["RED", "GRAY"],
            ["GRAY", "RED"],
            []
        ], capacities=3)

        moves = get_legal_reveal_moves(board)
        for m in moves:
            self.assertNotEqual(m.move.to_tube, 2)

    def test_no_legal_reveal_moves_returns_none(self):
        # Stalled state: all revealed top balls have nowhere legal to go
        board = Board.from_lists([
            ["RED", "GRAY"],
            ["BLUE", "GRAY"]
        ], capacities=2)

        best_move = select_best_reveal_move(board)
        self.assertIsNone(best_move)


class TestBonusVerifier(unittest.TestCase):
    def test_successful_structural_verification(self):
        b_before = Board.from_lists([
            ["PINK", "GRAY", "GRAY"],
            ["PINK"],
            []
        ], capacities=3)

        # After move: Tube 1 -> Tube 2 | PINK x1
        # The previously GRAY ball in Tube 1 is now revealed as YELLOW
        b_after = Board.from_lists([
            ["YELLOW", "GRAY"],
            ["PINK", "PINK"],
            []
        ], capacities=3)

        move = Move(from_tube=1, to_tube=2, ball_count=1, color="PINK")
        ok, msg = verify_reveal_transition(b_before, b_after, move, expected_to_expose=True)
        self.assertTrue(ok, msg)

    def test_verification_fails_if_ball_count_violated(self):
        b_before = Board.from_lists([
            ["PINK", "GRAY"],
            []
        ], capacities=2)

        # Corrupted board after move: lost a ball
        b_after = Board.from_lists([
            ["YELLOW"],
            []
        ], capacities=2)

        move = Move(from_tube=1, to_tube=2, ball_count=1, color="PINK")
        ok, msg = verify_reveal_transition(b_before, b_after, move, expected_to_expose=True)
        self.assertFalse(ok)
        self.assertIn("Ball count violated", msg)

    def test_verification_fails_if_mystery_count_does_not_decrease(self):
        b_before = Board.from_lists([
            ["PINK", "GRAY"],
            []
        ], capacities=2)

        # Tube 1 still perceived as GRAY on top after move
        b_after = Board.from_lists([
            ["GRAY"],
            ["PINK"]
        ], capacities=2)

        move = Move(from_tube=1, to_tube=2, ball_count=1, color="PINK")
        ok, msg = verify_reveal_transition(b_before, b_after, move, expected_to_expose=True)
        self.assertFalse(ok)
        self.assertIn("still perceived as GRAY", msg)


class TestBonusController(unittest.TestCase):
    def test_stalled_reveal_loop_aborts_safely(self):
        stalled_board = Board.from_lists([
            ["RED", "GRAY"],
            ["BLUE", "GRAY"]
        ], capacities=2)

        cfg = AutomationConfig(dry_run=True)
        report = run_bonus_reveal_loop(stalled_board, vision_tubes=[], emulator=None, config=cfg)
        self.assertFalse(report.success)
        self.assertIsNotNone(report.abort_reason)
        self.assertIn("STALLED", report.abort_reason)

    def test_dry_run_reveal_loop_execution(self):
        board = Board.from_lists([
            ["PINK", "GRAY"],
            ["PINK", "GRAY"],
            [],
            []
        ], capacities=2)

        cfg = AutomationConfig(dry_run=True)
        report = run_bonus_reveal_loop(board, vision_tubes=[], emulator=None, config=cfg, max_iterations=5)
        self.assertGreater(report.total_iterations, 0)

    def test_partially_filled_destination_board_and_planning_regression(self):
        # State after Move 1: Tube 7 received single YELLOW ball at bottom
        board = Board.from_lists([
            ["GREEN", "GREEN", "GRAY", "GRAY"],
            ["YELLOW", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["YELLOW"]
        ], capacities=4)

        self.assertTrue(board.has_mystery_balls)
        self.assertEqual(board.total_balls, 20)
        self.assertEqual(board.mystery_ball_count, 13)
        self.assertEqual(board.known_ball_count, 7)
        self.assertEqual(board.get_tube(7).balls, ["YELLOW"])
        self.assertEqual(board.get_tube(7).available_space, 3)

        best_move = select_best_reveal_move(board)
        self.assertIsNotNone(best_move)
        # Should merge Tube 2 (YELLOW) onto Tube 7 (YELLOW)
        self.assertEqual(best_move.move.from_tube, 2)
        self.assertEqual(best_move.move.to_tube, 7)
        self.assertEqual(best_move.move.color, "YELLOW")
        self.assertTrue(best_move.will_expose_mystery)
        self.assertEqual(best_move.priority, 20)


class TestBonusSynchronization(unittest.TestCase):
    def test_reveal_context_freezes_state_and_guarantees_alignment(self):
        from models.tube import Tube
        from bonus.models import RevealStepContext
        from bonus.controller import create_reveal_context

        board = Board.from_lists([
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)

        # Mock Tube objects
        tubes = [
            Tube(id=1, x=55, y=285, width=84, height=290, area=24360, contour=None),
            Tube(id=2, x=223, y=285, width=83, height=290, area=24070, contour=None),
            Tube(id=3, x=139, y=689, width=83, height=290, area=24070, contour=None),
            Tube(id=4, x=306, y=689, width=84, height=290, area=24360, contour=None),
        ]

        reveal_move = select_best_reveal_move(board)
        self.assertIsNotNone(reveal_move)

        ctx = create_reveal_context(board, tubes, reveal_move)
        self.assertIsInstance(ctx, RevealStepContext)
        self.assertEqual(ctx.source_tube_id, reveal_move.move.from_tube)
        self.assertEqual(ctx.destination_tube_id, reveal_move.move.to_tube)
        self.assertEqual(ctx.transferred_color, reveal_move.move.color)
        self.assertEqual(ctx.transfer_count, reveal_move.move.ball_count)
        self.assertEqual(ctx.will_expose_mystery, reveal_move.will_expose_mystery)

    def test_verifier_rejects_stale_destination_mismatch(self):
        from bonus.models import RevealStepContext

        # Pre-move board where Tube 1 -> Tube 3 was planned
        b_before = Board.from_lists([
            ["YELLOW", "GRAY"],
            [],
            []
        ], capacities=2)

        move = Move(from_tube=1, to_tube=3, ball_count=1, color="YELLOW")
        rm = RevealMove(move=move, will_expose_mystery=True, priority=10, reason="Test")

        ctx = RevealStepContext(
            before_board=b_before,
            reveal_move=rm,
            source_tube_id=1,
            destination_tube_id=3,
            transferred_color="YELLOW",
            transfer_count=1,
            source_tap_point=(100, 200),
            dest_tap_point=(300, 400),
            will_expose_mystery=True
        )

        # Actual post-move board where ball mistakenly went to Tube 2 instead of Tube 3
        b_after_mismatched = Board.from_lists([
            ["GREEN"],
            ["YELLOW"],  # Ball in Tube 2!
            []           # Tube 3 is empty
        ], capacities=2)

        # Verifier using frozen context MUST detect that Tube 3 did not receive the ball
        ok, msg = verify_reveal_transition(ctx, b_after_mismatched)
        self.assertFalse(ok)
        self.assertIn("Destination Tube 3 top color is None, expected YELLOW", msg)

    def test_post_reveal_1_state_passes_for_correct_transition(self):
        # Pre-move board: Tube 1 had YELLOW, Tube 7 was empty
        b_before = Board.from_lists([
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)

        # Move: Tube 1 -> Tube 7 | YELLOW x1
        move = Move(from_tube=1, to_tube=7, ball_count=1, color="YELLOW")
        rm = RevealMove(move=move, will_expose_mystery=True, priority=10, reason="Exposes mystery")

        ctx = RevealStepContext(
            before_board=b_before,
            reveal_move=rm,
            source_tube_id=1,
            destination_tube_id=7,
            transferred_color="YELLOW",
            transfer_count=1,
            source_tap_point=(97, 444),
            dest_tap_point=(515, 848),
            will_expose_mystery=True
        )

        # Post-move state after YELLOW moved to Tube 7 and Tube 1 exposed GREEN (the actual real state)
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["YELLOW"]
        ], capacities=4)

        ok, msg = verify_reveal_transition(ctx, b_after)
        self.assertTrue(ok, msg)

    def test_full_reveal_loop_transitions_to_canonical_bfs(self):
        from solver import solve

        # Initial bonus board with 2 mystery balls
        initial_board = Board.from_lists([
            ["PINK", "GRAY"],
            ["PINK", "GRAY"],
            [],
            []
        ], capacities=2)

        cfg = AutomationConfig(dry_run=True, tap_delay=0.85, bonus_tap_delay=0.85)
        report = run_bonus_reveal_loop(initial_board, vision_tubes=[], emulator=None, config=cfg, max_iterations=5)

        self.assertTrue(report.success)
        self.assertIsNotNone(report.final_board)
        self.assertEqual(report.final_board.mystery_ball_count, 0)
        self.assertFalse(report.final_board.has_mystery_balls)

        # Handoff to Canonical BFS
        sol = solve(report.final_board)
        self.assertTrue(sol.solved)
        self.assertGreaterEqual(len(sol.moves), 0)

    def test_reveal_loop_stops_on_planner_stall(self):
        # Unsolvable / stall board with no empty spaces and no matching top colors
        stall_board = Board.from_lists([
            ["PINK", "GRAY"],
            ["YELLOW", "GRAY"],
        ], capacities=2)

        cfg = AutomationConfig(dry_run=True)
        report = run_bonus_reveal_loop(stall_board, vision_tubes=[], emulator=None, config=cfg, max_iterations=5)
        self.assertFalse(report.success)
        self.assertIn("STALLED", report.abort_reason)


class TestBonusEmptyReserveGroupVerification(unittest.TestCase):
    def setUp(self):
        # 7-tube initial bonus board with 2 empty reserve tubes (Tube 6 and Tube 7)
        self.board_before = Board.from_lists([
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)

        # Move: Tube 1 -> Tube 6 | YELLOW x1
        self.reveal_move = RevealMove(
            move=Move(from_tube=1, to_tube=6, ball_count=1, color="YELLOW"),
            will_expose_mystery=True,
            priority=10,
            reason="Exposes mystery into empty reserve"
        )

        self.ctx = RevealStepContext(
            before_board=self.board_before,
            reveal_move=self.reveal_move,
            source_tube_id=1,
            destination_tube_id=6,
            transferred_color="YELLOW",
            transfer_count=1,
            source_tap_point=(97, 444),
            dest_tap_point=(348, 848),
            will_expose_mystery=True,
            is_empty_reserve_move=True,
            valid_empty_reserve_ids=(6, 7)
        )

    def test_empty_reserve_physical_transfer_into_tube_7_passes(self):
        # Physical quirk: YELLOW landed in Tube 7 while Tube 6 remained empty
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["YELLOW"]
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertTrue(ok, f"Expected PASS for valid interchangeable empty reserve transfer, got {msg}")

    def test_empty_reserve_transfer_into_planned_tube_6_passes(self):
        # Normal ideal landing: YELLOW landed directly in Tube 6
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["YELLOW"],
            []
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertTrue(ok, f"Expected PASS when ball lands directly in planned Tube 6, got {msg}")

    def test_empty_reserve_wrong_color_fails(self):
        # Wrong color: RED appeared in Tube 7 instead of YELLOW
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["RED"]
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertFalse(ok)
        self.assertIn("expected YELLOW", msg)

    def test_empty_reserve_ball_duplication_fails(self):
        # Ball duplication: Both Tube 6 and Tube 7 received YELLOW balls
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["YELLOW"],
            ["YELLOW"]
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertFalse(ok)

    def test_empty_reserve_ball_disappeared_fails(self):
        # Ball disappeared: Neither Tube 6 nor Tube 7 received the ball
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertFalse(ok)

    def test_empty_reserve_unrelated_ball_appears_fails(self):
        # Unrelated ball appeared in the other reserve
        b_after = Board.from_lists([
            ["GREEN", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["YELLOW"],
            ["GREEN"]
        ], capacities=4)

        ok, msg = verify_reveal_transition(self.ctx, b_after)
        self.assertFalse(ok)

    def test_occupied_destination_strictly_enforces_exact_tube(self):
        # Occupied destination: Tube 2 -> Tube 7 (which already had YELLOW)
        # MUST strictly require Tube 7, and cannot land in Tube 6
        b_occ_before = Board.from_lists([
            ["GREEN", "GREEN", "GRAY", "GRAY"],
            ["YELLOW", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["YELLOW"]
        ], capacities=4)

        rm_occ = RevealMove(
            move=Move(from_tube=2, to_tube=7, ball_count=1, color="YELLOW"),
            will_expose_mystery=True,
            priority=20,
            reason="Merges with matching color"
        )

        ctx_occ = RevealStepContext(
            before_board=b_occ_before,
            reveal_move=rm_occ,
            source_tube_id=2,
            destination_tube_id=7,
            transferred_color="YELLOW",
            transfer_count=1,
            source_tap_point=(264, 444),
            dest_tap_point=(515, 848),
            will_expose_mystery=True,
            is_empty_reserve_move=False,
            valid_empty_reserve_ids=()
        )

        # If it landed in Tube 6 instead of Tube 7, it MUST FAIL
        b_occ_mismatched = Board.from_lists([
            ["GREEN", "GREEN", "GRAY", "GRAY"],
            ["YELLOW", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["YELLOW"],
            ["YELLOW"]
        ], capacities=4)
        ok, msg = verify_reveal_transition(ctx_occ, b_occ_mismatched)
        self.assertFalse(ok)
        self.assertIn("Destination Tube 7", msg)

    def test_timing_defaults_bonus_0_50_and_normal_0_35(self):
        cfg = AutomationConfig()
        self.assertEqual(cfg.tap_delay, 0.35, "Normal tap delay must be 0.35s")
        self.assertEqual(cfg.bonus_tap_delay, 0.50, "Bonus tap delay must be 0.50s")
        self.assertEqual(cfg.bonus_move_settle_delay, 1.20, "Bonus settle delay must be 1.20s")
        self.assertEqual(cfg.auto_empty_settle_delay, 1.50, "Auto-empty settle delay must be 1.50s")

    def test_normal_level_destination_verification_unchanged(self):
        # In a normal level without mystery balls, occupied destination mismatch must fail
        from automation.verifier import compare_boards
        b_before = Board.from_lists([["RED", "BLUE"], ["BLUE"]], capacities=2)
        b_expected = Board.from_lists([["RED"], ["BLUE", "BLUE"]], capacities=2)
        b_wrong = Board.from_lists([["RED"], ["BLUE"]], capacities=2)
        match, diffs = compare_boards(b_expected, b_wrong)
        self.assertFalse(match)
        self.assertTrue(len(diffs) > 0)


class TestBonusMainRouting(unittest.TestCase):
    def test_multi_iteration_reveal_loop_full_pipeline(self):
        # 4-tube bonus board with 2 mystery balls
        initial_board = Board.from_lists([
            ["PINK", "GRAY"],
            ["PINK", "GRAY"],
            [],
            []
        ], capacities=2)

        cfg = AutomationConfig(dry_run=True, tap_delay=0.50, bonus_tap_delay=0.50)
        report = run_bonus_reveal_loop(initial_board, vision_tubes=[], emulator=None, config=cfg, max_iterations=10)

        self.assertTrue(report.success)
        self.assertIsNotNone(report.final_board)
        self.assertEqual(report.final_board.mystery_ball_count, 0)
        self.assertEqual(report.total_iterations, 2)

        # Hand off to Canonical BFS solver
        from solver import solve
        sol = solve(report.final_board)
        self.assertTrue(sol.solved)


class TestBonusLifecycleAndSequencing(unittest.TestCase):
    """
    Regression tests for execution order and atomic move lifecycle.
    Guarantees Move N is fully captured and verified before Move N+1 can touch the screen.
    """
    def test_verification_must_precede_next_move_dispatch(self):
        events = []

        class EventTrackingEmulator:
            def __init__(self):
                self.call_count = 0

            def run(self, command):
                if len(command) >= 4 and command[0] == "shell" and command[1] == "input" and command[2] == "tap":
                    events.append(("tap", int(command[3]), int(command[4])))
                return "OK", ""

            def capture_screenshot(self, path):
                events.append(("capture_screenshot", path))
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                return dummy, path

        # 4-tube initial board
        initial_board = Board.from_lists([
            ["PINK", "GRAY"],
            ["PINK", "GRAY"],
            [],
            []
        ], capacities=2)

        tube1 = Tube(id=1, x=100, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube2 = Tube(id=2, x=220, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube3 = Tube(id=3, x=340, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube4 = Tube(id=4, x=460, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tubes = [tube1, tube2, tube3, tube4]

        # Iteration 1 after-board (revealed PINK in Tube 1, Tube 3 got PINK via AUTO_EMPTY)
        board_iter1 = Board.from_lists([
            ["PINK"],
            ["PINK", "GRAY"],
            ["PINK"],
            []
        ], capacities=2)

        # Iteration 2 after-board (revealed PINK in Tube 2, Tube 1 got 2nd PINK via EXACT_TUBE)
        board_iter2 = Board.from_lists([
            ["PINK", "PINK"],
            ["PINK"],
            ["PINK"],
            []
        ], capacities=2)

        iteration_boards = [board_iter1, board_iter2]
        iter_idx = [0]

        def mock_vision_pipeline(img, *args, **kwargs):
            events.append(("vision_pipeline", iter_idx[0]))
            from vision.pipeline import VisionResult
            b = iteration_boards[iter_idx[0]]
            iter_idx[0] += 1
            return VisionResult(image=img, tubes=tubes, board_state=[list(t.balls) for t in b.tubes], board=b)

        em = EventTrackingEmulator()
        cfg = AutomationConfig(dry_run=False, tap_delay=0.01, bonus_tap_delay=0.01, move_settle_delay=0.01)

        with patch("bonus.controller.run_bonus_vision", side_effect=mock_vision_pipeline):
            report = run_bonus_reveal_loop(initial_board, vision_tubes=tubes, emulator=em, config=cfg, max_iterations=2)

        self.assertTrue(report.success)
        self.assertEqual(report.total_iterations, 2)

        # Trace event sequence
        # Iteration 1 (AUTO_EMPTY): tap(src) -> capture_screenshot -> vision_pipeline
        # Iteration 2 (EXACT_TUBE): tap(src) -> tap(dst) -> capture_screenshot -> vision_pipeline
        event_names = [e[0] for e in events]
        self.assertEqual(
            event_names,
            [
                "tap", "capture_screenshot", "vision_pipeline",
                "tap", "tap", "capture_screenshot", "vision_pipeline"
            ],
            f"Event sequence must strictly interleave tap -> capture -> vision before next taps: {event_names}"
        )

    def test_failed_verification_strictly_halts_and_blocks_subsequent_taps(self):
        events = []

        class EventTrackingEmulator:
            def run(self, command):
                if len(command) >= 4 and command[0] == "shell" and command[1] == "input" and command[2] == "tap":
                    events.append(("tap", int(command[3]), int(command[4])))
                return "OK", ""

            def capture_screenshot(self, path):
                events.append(("capture_screenshot", path))
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                return dummy, path

        # 4-tube board
        initial_board = Board.from_lists([
            ["LIGHT_BLUE", "GRAY"],
            ["GREEN", "GRAY"],
            [],
            []
        ], capacities=2)

        tube1 = Tube(id=1, x=100, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube2 = Tube(id=2, x=220, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube3 = Tube(id=3, x=340, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube4 = Tube(id=4, x=460, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tubes = [tube1, tube2, tube3, tube4]

        # Perceived board after move 1 is missing a ball (e.g. 19 balls / 3 balls instead of 4)
        mismatched_board = Board.from_lists([
            ["GRAY"],
            ["GREEN", "GRAY"],
            [],
            []
        ], capacities=2)

        def mock_vision_pipeline(img, *args, **kwargs):
            events.append(("vision_pipeline", "mismatch"))
            from vision.pipeline import VisionResult
            return VisionResult(image=img, tubes=tubes, board_state=[list(t.balls) for t in mismatched_board.tubes], board=mismatched_board)

        em = EventTrackingEmulator()
        cfg = AutomationConfig(dry_run=False, tap_delay=0.01, bonus_tap_delay=0.01, move_settle_delay=0.01)

        with patch("bonus.controller.run_bonus_vision", side_effect=mock_vision_pipeline):
            report = run_bonus_reveal_loop(initial_board, vision_tubes=tubes, emulator=em, config=cfg, max_iterations=5)

        self.assertFalse(report.success)
        self.assertIn("Verification failed", report.abort_reason)

        # Ensure that after the verification failure for AUTO_EMPTY (1 tap), NO further taps were dispatched!
        taps_sent = [e for e in events if e[0] == "tap"]
        self.assertEqual(len(taps_sent), 1, "Failed verification must block all subsequent taps immediately.")

    def test_unauthorized_physical_tap_during_verification_is_detected_and_aborted(self):
        class SpuriousTapEmulator:
            def run(self, command):
                return "OK", ""

            def capture_screenshot(self, path):
                # Simulate a stray tap occurring before screenshot is returned!
                tap(300, 300, emulator=self, purpose="UNAUTHORIZED_STRAY_TAP")
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                return dummy, path

        initial_board = Board.from_lists([
            ["LIGHT_BLUE", "GRAY"],
            ["GREEN", "GRAY"],
            [],
            []
        ], capacities=2)

        tube1 = Tube(id=1, x=100, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube2 = Tube(id=2, x=220, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube3 = Tube(id=3, x=340, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tube4 = Tube(id=4, x=460, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
        tubes = [tube1, tube2, tube3, tube4]

        em = SpuriousTapEmulator()
        cfg = AutomationConfig(dry_run=False, tap_delay=0.01, bonus_tap_delay=0.01, move_settle_delay=0.01)

        report = run_bonus_reveal_loop(initial_board, vision_tubes=tubes, emulator=em, config=cfg, max_iterations=2)
        self.assertFalse(report.success)
        self.assertIn("UNAUTHORIZED PHYSICAL INPUT DETECTED", report.abort_reason)

    def test_iteration_8_auto_empty_transfer_dispatches_exactly_one_tap(self):
        """
        Regression Test for Iteration 8 scenario:
        Tube 1: [RED, RED]
        Tube 2: [GREEN, GREEN, GREEN, GREEN]
        Tube 3: [LIGHT_BLUE, GRAY, GRAY, GRAY]
        Tube 4: [PINK, GRAY, GRAY, GRAY]
        Tube 5: [PINK, GRAY, GRAY, GRAY]
        Tube 6: []
        Tube 7: [YELLOW, YELLOW, YELLOW]

        Move selected: Tube 3 -> Tube 6 | LIGHT_BLUE x1
        Because there is no occupied LIGHT_BLUE tube, destination_mode MUST be AUTO_EMPTY.
        Physical dispatch MUST send exactly 1 tap (source tap ONLY, zero destination taps).
        """
        events = []

        class SingleTapTrackingEmulator:
            def run(self, command):
                if len(command) >= 4 and command[0] == "shell" and command[1] == "input" and command[2] == "tap":
                    events.append(("tap", int(command[3]), int(command[4])))
                return "OK", ""

            def capture_screenshot(self, path):
                events.append(("capture_screenshot", path))
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                return dummy, path

        initial_board = Board.from_lists([
            ["RED", "RED"],
            ["GREEN", "GREEN", "GREEN", "GREEN"],
            ["LIGHT_BLUE", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            ["YELLOW", "YELLOW", "YELLOW"]
        ], capacities=4)

        tubes = [
            Tube(id=i, x=100 * i, y=200, width=80, height=300, area=24000.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
            for i in range(1, 8)
        ]

        # Verify classification in create_reveal_context
        reveal_move = select_best_reveal_move(initial_board)
        self.assertIsNotNone(reveal_move)
        self.assertEqual(reveal_move.move.from_tube, 3)
        self.assertEqual(reveal_move.move.color, "LIGHT_BLUE")

        ctx = create_reveal_context(initial_board, tubes, reveal_move)
        self.assertEqual(ctx.destination_mode, DestinationMode.AUTO_EMPTY, "Must be classified as AUTO_EMPTY")

        # Mock post-move board where game automatically placed LIGHT_BLUE in Tube 6
        after_board = Board.from_lists([
            ["RED", "RED"],
            ["GREEN", "GREEN", "GREEN", "GREEN"],
            ["PINK", "GRAY", "GRAY"],  # Revealed PINK in Tube 3
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["LIGHT_BLUE"],             # Auto-placed in Tube 6
            ["YELLOW", "YELLOW", "YELLOW"]
        ], capacities=4)

        def mock_vision_pipeline(img, *args, **kwargs):
            events.append(("vision_pipeline", "iteration_8"))
            from vision.pipeline import VisionResult
            return VisionResult(image=img, tubes=tubes, board_state=[list(t.balls) for t in after_board.tubes], board=after_board)

        em = SingleTapTrackingEmulator()
        cfg = AutomationConfig(
            dry_run=False,
            bonus_tap_delay=0.01,
            bonus_move_settle_delay=0.01,
            auto_empty_settle_delay=0.01
        )

        with patch("bonus.controller.run_bonus_vision", side_effect=mock_vision_pipeline):
            ok, msg, new_b, ctx_out = run_bonus_reveal_single_step(initial_board, tubes, emulator=em, config=cfg)

        self.assertTrue(ok, f"Expected PASS for AUTO_EMPTY reveal move, got: {msg}")
        taps_sent = [e for e in events if e[0] == "tap"]
        self.assertEqual(len(taps_sent), 1, f"AUTO_EMPTY must dispatch EXACTLY ONE tap (source tap), found {len(taps_sent)}")
        self.assertEqual(taps_sent[0], ("tap", 340, 365), "Must be source tube tap coordinates")


class TestBonusCompletedTubeDetection(unittest.TestCase):
    """
    Regression tests proving that completed / corked / celebrated tubes are NEVER
    dropped from the board because stable initial geometry is frozen and reused.
    """
    def test_completed_corked_tube_preserved_by_stable_geometry(self):
        # 7-tube bonus layout (5 top, 2 bottom)
        initial_tubes = [
            Tube(id=i, x=50 + (i - 1) * 90, y=200 if i <= 5 else 600, width=70, height=280, area=19600.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
            for i in range(1, 8)
        ]

        # Synthetic screenshot
        img = np.zeros((1280, 720, 3), dtype=np.uint8)

        # Mock detect_tube_occupancy and get_ball_color/classify_color
        def mock_occupancy(image, tube):
            slots = [(tube.x + 35, tube.y + 50 + j * 55) for j in range(4)]
            if tube.id == 2:
                # Tube 2 is completed GREEN tube (4 balls + cork)
                return slots, [True, True, True, True]
            elif tube.id == 1:
                return slots, [True, True, False, False]
            elif tube.id in (3, 4, 5):
                return slots, [True, True, True, True]
            elif tube.id == 7:
                return slots, [True, True, False, False]
            else: # Tube 6 is empty
                return slots, [False, False, False, False]

        def mock_classify(h, s, v):
            return "GREEN"

        with patch("vision.pipeline.detect_tube_occupancy", side_effect=mock_occupancy), \
             patch("vision.pipeline.classify_color", return_value="GREEN"):
            res = run_bonus_vision(img, stable_tubes=initial_tubes, save_debug=False)

        self.assertEqual(res.total_tubes, 7, "All 7 tubes must remain present; completed tube cannot disappear")
        self.assertEqual(res.total_balls, 20)
        tube2 = res.board.get_tube(2)
        self.assertEqual(tube2.balls, ["GREEN", "GREEN", "GREEN", "GREEN"])
        self.assertTrue(tube2.is_solved)

    def test_completed_top_row_and_bottom_row_tubes_preserved(self):
        initial_tubes = [
            Tube(id=i, x=50 + (i - 1) * 90, y=200 if i <= 5 else 600, width=70, height=280, area=19600.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
            for i in range(1, 8)
        ]
        img = np.zeros((1280, 720, 3), dtype=np.uint8)

        def mock_occupancy(image, tube):
            slots = [(tube.x + 35, tube.y + 50 + j * 55) for j in range(4)]
            if tube.id in (1, 6): # Top row Tube 1 and Bottom row Tube 6 completed
                return slots, [True, True, True, True]
            return slots, [False, False, False, False]

        with patch("vision.pipeline.detect_tube_occupancy", side_effect=mock_occupancy), \
             patch("vision.pipeline.classify_color", return_value="RED"):
            res = run_bonus_vision(img, stable_tubes=initial_tubes, save_debug=False)

        self.assertEqual(res.total_tubes, 7)
        self.assertEqual(res.board.get_tube(1).balls, ["RED", "RED", "RED", "RED"])
        self.assertEqual(res.board.get_tube(6).balls, ["RED", "RED", "RED", "RED"])

    def test_bonus_vision_skips_contour_discovery_and_executes_fast(self):
        initial_tubes = [
            Tube(id=i, x=50 + (i - 1) * 90, y=200 if i <= 5 else 600, width=70, height=280, area=19600.0, contour=np.zeros((4, 1, 2), dtype=np.int32))
            for i in range(1, 8)
        ]
        img = np.zeros((1280, 720, 3), dtype=np.uint8)

        with patch("vision.pipeline.detect_tubes") as mock_detect:
            res = run_bonus_vision(img, stable_tubes=initial_tubes, save_debug=False)
            mock_detect.assert_not_called()
            self.assertEqual(res.total_tubes, 7)


if __name__ == "__main__":
    unittest.main()
