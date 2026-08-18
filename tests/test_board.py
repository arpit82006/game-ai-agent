"""
Unit tests for the formal Board and TubeState models.
Tests logic, invariants, operations, and validation independent of OpenCV.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board import Board, TubeState


class TestTubeState(unittest.TestCase):
    def test_empty_tube(self):
        tube = TubeState(id=1, capacity=4, balls=[])
        self.assertTrue(tube.is_empty)
        self.assertFalse(tube.is_full)
        self.assertEqual(tube.ball_count, 0)
        self.assertEqual(tube.available_space, 4)
        self.assertIsNone(tube.top_color)
        self.assertEqual(tube.top_same_color_count, 0)
        self.assertTrue(tube.is_solved)

    def test_partially_filled_tube(self):
        tube = TubeState(id=2, capacity=4, balls=["RED", "BLUE"])
        self.assertFalse(tube.is_empty)
        self.assertFalse(tube.is_full)
        self.assertEqual(tube.ball_count, 2)
        self.assertEqual(tube.available_space, 2)
        self.assertEqual(tube.top_color, "RED")
        self.assertEqual(tube.top_same_color_count, 1)
        self.assertFalse(tube.is_solved)

    def test_full_tube_monochromatic(self):
        tube = TubeState(id=3, capacity=4, balls=["RED", "RED", "RED", "RED"])
        self.assertFalse(tube.is_empty)
        self.assertTrue(tube.is_full)
        self.assertTrue(tube.is_pure)
        self.assertTrue(tube.is_solved)
        self.assertEqual(tube.top_color, "RED")
        self.assertEqual(tube.top_same_color_count, 4)
        self.assertEqual(tube.available_space, 0)

    def test_full_tube_mixed_colors(self):
        tube = TubeState(id=4, capacity=4, balls=["RED", "RED", "BLUE", "BLUE"])
        self.assertTrue(tube.is_full)
        self.assertFalse(tube.is_pure)
        self.assertFalse(tube.is_solved)
        self.assertEqual(tube.top_color, "RED")
        self.assertEqual(tube.top_same_color_count, 2)

    def test_push_and_pop_stack_order(self):
        tube = TubeState(id=1, capacity=4, balls=["BLUE", "GREEN"])
        # Top ball is BLUE
        self.assertEqual(tube.top_color, "BLUE")
        # Push RED to top
        tube.push("RED")
        self.assertEqual(tube.balls, ["RED", "BLUE", "GREEN"])
        self.assertEqual(tube.top_color, "RED")
        self.assertEqual(tube.ball_count, 3)
        # Pop top ball
        popped = tube.pop()
        self.assertEqual(popped, "RED")
        self.assertEqual(tube.balls, ["BLUE", "GREEN"])

    def test_push_over_capacity_raises(self):
        tube = TubeState(id=1, capacity=2, balls=["RED", "BLUE"])
        with self.assertRaises(ValueError):
            tube.push("GREEN")

    def test_pop_empty_raises(self):
        tube = TubeState(id=1, capacity=4, balls=[])
        with self.assertRaises(ValueError):
            tube.pop()


class TestBoardModel(unittest.TestCase):
    def test_from_lists_and_indexing(self):
        raw = [
            ["RED", "YELLOW", "BLUE", "RED"],
            ["GREEN", "GREEN"],
            [],
            []
        ]
        board = Board.from_lists(raw, capacities=[4, 3, 4, 4])
        self.assertEqual(board.num_tubes, 4)
        self.assertEqual(board.total_balls, 6)
        self.assertEqual(board.empty_tubes_count, 2)
        self.assertEqual(board.get_tube(1).capacity, 4)
        self.assertEqual(board.get_tube(2).capacity, 3)
        self.assertEqual(board[0].balls, ["RED", "YELLOW", "BLUE", "RED"])
        self.assertEqual(board[1].balls, ["GREEN", "GREEN"])
        self.assertEqual(board[2].balls, [])
        self.assertEqual(board[3].balls, [])

    def test_safe_copy_independence(self):
        b1 = Board.from_lists([["RED", "BLUE"], ["GREEN"], []])
        b2 = b1.copy()

        # Modify copy
        b2[0].pop()
        b2[1].push("YELLOW")

        # Verify original is unmutated
        self.assertEqual(b1[0].balls, ["RED", "BLUE"])
        self.assertEqual(b1[1].balls, ["GREEN"])
        self.assertEqual(b2[0].balls, ["BLUE"])
        self.assertEqual(b2[1].balls, ["YELLOW", "GREEN"])

    def test_to_lists_and_to_dict(self):
        raw = [["RED", "BLUE"], ["GREEN"], []]
        board = Board.from_lists(raw, capacities=3)
        self.assertEqual(board.to_lists(), [["RED", "BLUE"], ["GREEN"], []])

        d = board.to_dict()
        self.assertEqual(d["num_tubes"], 3)
        self.assertEqual(d["total_balls"], 3)
        self.assertEqual(d["empty_tubes"], 1)
        self.assertEqual(len(d["tubes"]), 3)

    def test_hashable_state_tuple(self):
        b1 = Board.from_lists([["RED", "BLUE"], []])
        b2 = Board.from_lists([["RED", "BLUE"], []])
        s1 = b1.to_state_tuple()
        s2 = b2.to_state_tuple()
        self.assertEqual(s1, s2)
        visited = {s1}
        self.assertIn(s2, visited)

    def test_is_solved_board(self):
        solved_board = Board.from_lists([
            ["RED", "RED", "RED", "RED"],
            ["BLUE", "BLUE", "BLUE", "BLUE"],
            [],
            []
        ], capacities=4)
        self.assertTrue(solved_board.is_solved)

        unsolved_board = Board.from_lists([
            ["RED", "RED", "RED", "BLUE"],
            ["BLUE", "BLUE", "BLUE", "RED"],
            [],
            []
        ], capacities=4)
        self.assertFalse(unsolved_board.is_solved)


class TestBoardValidation(unittest.TestCase):
    def test_valid_board_passes(self):
        board = Board.from_lists([
            ["ORANGE", "MAGENTA", "LIGHT_BLUE", "ORANGE"],
            ["DARK_PURPLE", "EMERALD_GREEN", "EMERALD_GREEN", "LIGHT_BLUE"],
            ["ORANGE", "DARK_PURPLE", "DARK_PURPLE", "LIGHT_BLUE"],
            ["MAGENTA", "EMERALD_GREEN", "LIGHT_BLUE", "ORANGE"],
            ["MAGENTA", "MAGENTA", "DARK_PURPLE", "EMERALD_GREEN"],
            [],
            []
        ], capacities=4)
        is_valid, errors = board.validate()
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_over_capacity_rejection(self):
        board = Board.from_lists([
            ["RED", "RED", "RED", "RED", "RED"],  # 5 balls in cap-4 tube
            []
        ], capacities=4)
        is_valid, errors = board.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeding capacity" in e for e in errors))

    def test_zero_or_negative_capacity_rejection(self):
        t1 = TubeState(id=1, capacity=0, balls=["RED"])
        board = Board(tubes=[t1])
        is_valid, errors = board.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("invalid capacity" in e for e in errors))

    def test_empty_board_rejection(self):
        board = Board.from_lists([[], []], capacities=4)
        is_valid, errors = board.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("zero balls" in e for e in errors))

    def test_zero_tubes_rejection(self):
        board = Board(tubes=[])
        is_valid, errors = board.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("zero tubes" in e for e in errors))

    def test_invalid_color_marker_rejection(self):
        t1 = TubeState(id=1, capacity=4, balls=["RED", "EMPTY", "BLUE"])
        board = Board(tubes=[t1])
        is_valid, errors = board.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("invalid color marker" in e for e in errors))

    def test_future_special_color_support(self):
        # A special level with a newly named color such as "NEON_CYAN"
        board = Board.from_lists([
            ["NEON_CYAN", "NEON_CYAN", "GOLD"],
            ["GOLD", "GOLD", "NEON_CYAN"],
            []
        ], capacities=3)
        is_valid, errors = board.validate()
        self.assertTrue(is_valid)
        self.assertIn("NEON_CYAN", board.colors)
        self.assertEqual(board.color_counts["NEON_CYAN"], 3)


if __name__ == "__main__":
    unittest.main()
