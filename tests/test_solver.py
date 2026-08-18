"""
Unit tests for the Ball Sort AI Solver subsystem.
Completely independent of OpenCV, BlueStacks, ADB, and live screenshots.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board import Board, TubeState
from solver import (
    Move,
    SolverResult,
    get_valid_moves,
    apply_move,
    solve,
    validate_and_replay_solution
)


class TestMoveGenerator(unittest.TestCase):
    def test_empty_source_generates_no_moves(self):
        board = Board.from_lists([[], ["RED", "BLUE"]], capacities=4)
        moves = get_valid_moves(board)
        # Cannot move from Tube 1 (empty)
        self.assertFalse(any(m.from_tube == 1 for m in moves))

    def test_full_destination_generates_no_moves(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["RED", "RED", "RED", "RED"]  # Full
        ], capacities=4)
        moves = get_valid_moves(board)
        # Cannot move into Tube 2 (full)
        self.assertFalse(any(m.to_tube == 2 for m in moves))

    def test_different_destination_color_is_rejected(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["GREEN", "YELLOW"]
        ], capacities=4)
        moves = get_valid_moves(board)
        # Cannot move RED onto GREEN, nor GREEN onto RED
        self.assertEqual(len(moves), 0)

    def test_same_destination_color_is_generated(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["RED", "GREEN"]
        ], capacities=4)
        moves = get_valid_moves(board)
        # Tube 1 -> Tube 2 (RED x1) and Tube 2 -> Tube 1 (RED x1)
        self.assertTrue(any(m.from_tube == 1 and m.to_tube == 2 and m.color == "RED" for m in moves))

    def test_empty_destination_is_generated(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            []
        ], capacities=4)
        moves = get_valid_moves(board)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0], Move(from_tube=1, to_tube=2, ball_count=1, color="RED"))

    def test_multiple_contiguous_balls_transfer_together(self):
        board = Board.from_lists([
            ["RED", "RED", "RED", "BLUE"],
            ["RED"],
            []
        ], capacities=4)
        moves = get_valid_moves(board)
        # Tube 1 has 3 contiguous REDs at top. Tube 2 has 1 RED and 3 empty slots.
        # Transfer should be min(3, 3) = 3 RED balls.
        move_t1_t2 = [m for m in moves if m.from_tube == 1 and m.to_tube == 2]
        self.assertEqual(len(move_t1_t2), 1)
        self.assertEqual(move_t1_t2[0].ball_count, 3)
        self.assertEqual(move_t1_t2[0].color, "RED")

    def test_destination_capacity_limits_transfer_count(self):
        board = Board.from_lists([
            ["RED", "RED", "RED", "BLUE"],
            ["RED", "RED", "GREEN"]  # Only 1 slot available in cap-4 tube
        ], capacities=4)
        moves = get_valid_moves(board)
        move_t1_t2 = [m for m in moves if m.from_tube == 1 and m.to_tube == 2]
        self.assertEqual(len(move_t1_t2), 1)
        # Destination only has 1 available slot, so only 1 RED ball transfers
        self.assertEqual(move_t1_t2[0].ball_count, 1)

    def test_source_and_destination_cannot_be_same_tube(self):
        board = Board.from_lists([["RED", "BLUE"], ["RED"]], capacities=4)
        moves = get_valid_moves(board)
        for m in moves:
            self.assertNotEqual(m.from_tube, m.to_tube)


class TestMoveApplication(unittest.TestCase):
    def test_single_ball_transfer(self):
        board = Board.from_lists([
            ["RED", "BLUE", "GREEN"],
            ["RED", "YELLOW"]
        ], capacities=4)
        move = Move(from_tube=1, to_tube=2, ball_count=1, color="RED")
        new_board = apply_move(board, move)

        # Verify source lost 1 RED
        self.assertEqual(new_board[0].balls, ["BLUE", "GREEN"])
        # Verify destination gained 1 RED at top
        self.assertEqual(new_board[1].balls, ["RED", "RED", "YELLOW"])
        # Verify original board is untouched
        self.assertEqual(board[0].balls, ["RED", "BLUE", "GREEN"])
        self.assertEqual(board[1].balls, ["RED", "YELLOW"])

    def test_multi_ball_transfer_preserves_ordering(self):
        board = Board.from_lists([
            ["RED", "RED", "BLUE"],
            []
        ], capacities=4)
        move = Move(from_tube=1, to_tube=2, ball_count=2, color="RED")
        new_board = apply_move(board, move)

        self.assertEqual(new_board[0].balls, ["BLUE"])
        self.assertEqual(new_board[1].balls, ["RED", "RED"])
        self.assertEqual(board[0].balls, ["RED", "RED", "BLUE"])
        self.assertEqual(board[1].balls, [])

    def test_illegal_move_raises_value_error(self):
        board = Board.from_lists([["RED", "BLUE"], ["GREEN", "YELLOW"]], capacities=4)
        # Move RED onto GREEN is illegal
        illegal_move = Move(from_tube=1, to_tube=2, ball_count=1, color="RED")
        with self.assertRaises(ValueError):
            apply_move(board, illegal_move)


class TestSolver(unittest.TestCase):
    def test_already_solved_board(self):
        board = Board.from_lists([
            ["RED", "RED", "RED", "RED"],
            ["BLUE", "BLUE", "BLUE", "BLUE"],
            []
        ], capacities=4)
        result = solve(board)
        self.assertTrue(result.solved)
        self.assertEqual(result.move_count, 0)
        self.assertEqual(result.moves, [])

    def test_one_move_solution(self):
        # Tube 1 has 1 RED on top of 3 BLUEs. Tube 2 has 3 REDs.
        board = Board.from_lists([
            ["RED", "BLUE", "BLUE", "BLUE"],
            ["RED", "RED", "RED"],
            ["BLUE"]
        ], capacities=4)
        result = solve(board)
        self.assertTrue(result.solved)
        self.assertGreaterEqual(result.move_count, 1)

        success, msg, final_board = validate_and_replay_solution(board, result.moves)
        self.assertTrue(success, msg)
        self.assertTrue(final_board.is_solved)

    def test_multi_move_solution_replay(self):
        # 3 colors, 5 tubes (2 empty)
        board = Board.from_lists([
            ["RED", "BLUE", "GREEN", "RED"],
            ["BLUE", "GREEN", "RED", "BLUE"],
            ["GREEN", "RED", "BLUE", "GREEN"],
            [],
            []
        ], capacities=4)
        result = solve(board)
        self.assertTrue(result.solved)
        self.assertGreater(result.move_count, 0)

        # Replay solution
        success, msg, final_board = validate_and_replay_solution(board, result.moves)
        self.assertTrue(success, msg)
        self.assertTrue(final_board.is_solved)

    def test_arbitrary_and_special_color_strings(self):
        # Test that solver works with novel color strings like "NEON_CYAN" or "GOLD_2"
        board = Board.from_lists([
            ["NEON_CYAN", "GOLD_2"],
            ["GOLD_2", "NEON_CYAN"],
            []
        ], capacities=2)
        result = solve(board)
        self.assertTrue(result.solved)

        success, msg, final_board = validate_and_replay_solution(board, result.moves)
        self.assertTrue(success, msg)
        self.assertTrue(final_board.is_solved)

    def test_mixed_capacities_handling(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["BLUE", "RED"],
            []
        ], capacities=[2, 2, 4])
        result = solve(board)
        self.assertTrue(result.solved)

        success, msg, final_board = validate_and_replay_solution(board, result.moves)
        self.assertTrue(success, msg)
        self.assertTrue(final_board.is_solved)

    def test_unsolvable_board_fails_gracefully(self):
        # Deadlocked board with no empty tubes and mismatching colors
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["BLUE", "RED"]
        ], capacities=2)
        result = solve(board)
        self.assertFalse(result.solved)
        self.assertIsNotNone(result.failure_reason)

    def test_replay_validator_detects_corrupted_move(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["BLUE", "RED"],
            []
        ], capacities=2)
        # Fabricate an invalid move (move GREEN when no green exists)
        bad_move = Move(from_tube=1, to_tube=3, ball_count=1, color="GREEN")
        success, msg, _ = validate_and_replay_solution(board, [bad_move])
        self.assertFalse(success)
        self.assertIn("Replay failed", msg)

    def test_ascii_board_rendering(self):
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["GREEN"],
            []
        ], capacities=3)
        from solver.visualizer import render_ascii_board
        rendered = render_ascii_board(board, title="Test Render")
        self.assertIn("T1", rendered)
        self.assertIn("T2", rendered)
        self.assertIn("T3", rendered)
        self.assertIn("RED", rendered)
        self.assertIn("BLU", rendered)
        self.assertIn("GRN", rendered)

    def test_replay_and_visualize_solution_conserves_balls_and_colors(self):
        from solver import replay_and_visualize_solution
        board = Board.from_lists([
            ["RED", "BLUE", "RED"],
            ["BLUE", "RED", "BLUE"],
            [],
            []
        ], capacities=3)
        result = solve(board)
        self.assertTrue(result.solved)

        # Run replay visualization with print_steps=False for fast test
        success, msg, history = replay_and_visualize_solution(board, result.moves, print_steps=False)
        self.assertTrue(success, msg)
        self.assertEqual(len(history), len(result.moves) + 1)

        # Verify conservation of ball counts and colors across EVERY intermediate step
        initial_balls = board.total_balls
        initial_colors = board.color_counts
        for idx, intermediate_board in enumerate(history):
            self.assertEqual(intermediate_board.total_balls, initial_balls, f"Ball count violated at step {idx}")
            self.assertEqual(intermediate_board.color_counts, initial_colors, f"Color counts violated at step {idx}")

        # Verify final step is solved
        self.assertTrue(history[-1].is_solved)

    def test_verbose_and_trace_solver_diagnostics(self):
        # Ensure verbose=True and trace=True run without throwing any exceptions
        board = Board.from_lists([
            ["RED", "BLUE"],
            ["BLUE", "RED"],
            []
        ], capacities=2)
        result = solve(board, verbose=True, trace=True, progress_interval=1)
        self.assertTrue(result.solved)


if __name__ == "__main__":
    unittest.main()

