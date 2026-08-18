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

    def test_variable_capacities_and_partial_fills_all_states(self):
        # Capacity 4: 0/4, 1/4, 2/4, 3/4, 4/4
        t_0_4 = TubeState(id=1, capacity=4, balls=[])
        t_1_4 = TubeState(id=2, capacity=4, balls=["PINK"])
        t_2_4 = TubeState(id=3, capacity=4, balls=["PINK", "GREEN"])
        t_3_4 = TubeState(id=4, capacity=4, balls=["PINK", "GREEN", "BLUE"])
        t_4_4 = TubeState(id=5, capacity=4, balls=["PINK", "GREEN", "BLUE", "RED"])

        self.assertEqual(t_0_4.ball_count, 0)
        self.assertEqual(t_1_4.ball_count, 1)
        self.assertEqual(t_2_4.ball_count, 2)
        self.assertEqual(t_3_4.ball_count, 3)
        self.assertEqual(t_4_4.ball_count, 4)

        # Capacity 3: 0/3, 1/3, 2/3, 3/3
        t_0_3 = TubeState(id=6, capacity=3, balls=[])
        t_1_3 = TubeState(id=7, capacity=3, balls=["PINK"])
        t_2_3 = TubeState(id=8, capacity=3, balls=["PINK", "GREEN"])
        t_3_3 = TubeState(id=9, capacity=3, balls=["PINK", "GREEN", "BLUE"])

        self.assertEqual(t_0_3.ball_count, 0)
        self.assertEqual(t_1_3.ball_count, 1)
        self.assertEqual(t_2_3.ball_count, 2)
        self.assertEqual(t_3_3.ball_count, 3)

        # Mixed capacity 8-tube board
        mixed_board = Board(tubes=[t_0_4, t_1_4, t_2_4, t_3_4, t_4_4, t_0_3, t_1_3, t_2_3])
    def test_bonus_level_mystery_balls_and_black_empty_semantics(self):
        # 1. Fully empty tube using BLACK markers
        b_empty = Board.from_lists([["BLACK", "BLACK", "BLACK", "BLACK"]], capacities=4)
        self.assertEqual(b_empty.tubes[0].ball_count, 0)
        self.assertTrue(b_empty.tubes[0].is_empty)

        # 2. One known ball + 3 gray mystery balls
        b_1k_3m = Board.from_lists([["PINK", "GRAY", "GRAY", "GRAY"]], capacities=4)
        self.assertEqual(b_1k_3m.tubes[0].ball_count, 4)
        self.assertEqual(b_1k_3m.tubes[0].balls, ["PINK", "GRAY", "GRAY", "GRAY"])
        self.assertTrue(b_1k_3m.has_mystery_balls)
        self.assertEqual(b_1k_3m.mystery_ball_count, 3)
        self.assertEqual(b_1k_3m.known_ball_count, 1)

        # 3. Two known balls + 2 gray mystery balls
        b_2k_2m = Board.from_lists([["PINK", "GREEN", "GRAY", "GRAY"]], capacities=4)
        self.assertEqual(b_2k_2m.tubes[0].ball_count, 4)
        self.assertEqual(b_2k_2m.tubes[0].balls, ["PINK", "GREEN", "GRAY", "GRAY"])

        # 4. Three known balls + 1 gray mystery ball
        b_3k_1m = Board.from_lists([["PINK", "GREEN", "RED", "GRAY"]], capacities=4)
        self.assertEqual(b_3k_1m.tubes[0].ball_count, 4)
        self.assertEqual(b_3k_1m.tubes[0].balls, ["PINK", "GREEN", "RED", "GRAY"])

        # 5. Fully known tube
        b_4k = Board.from_lists([["PINK", "GREEN", "RED", "YELLOW"]], capacities=4)
        self.assertEqual(b_4k.tubes[0].ball_count, 4)
        self.assertFalse(b_4k.has_mystery_balls)

        # 6. Fully mystery tube
        b_4m = Board.from_lists([["GRAY", "GRAY", "GRAY", "GRAY"]], capacities=4)
        self.assertEqual(b_4m.tubes[0].ball_count, 4)
        self.assertFalse(b_4m.tubes[0].is_pure)  # Mystery tubes cannot be considered pure
        self.assertFalse(b_4m.is_solved)

        # 7. Mixed known + mystery + empty states
        b_partial = Board.from_lists([["PINK", "GRAY", "BLACK", "BLACK"]], capacities=4)
        self.assertEqual(b_partial.tubes[0].ball_count, 2)
        self.assertEqual(b_partial.tubes[0].available_space, 2)
        self.assertEqual(b_partial.tubes[0].balls, ["PINK", "GRAY"])

        # 8. Full Bonus Level Board (5 mystery tubes + 2 empty tubes)
        bonus_board = Board.from_lists([
            ["YELLOW", "GRAY", "GRAY", "GRAY"],
            ["GREEN", "GRAY", "GRAY", "GRAY"],
            ["RED", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            ["PINK", "GRAY", "GRAY", "GRAY"],
            [],
            []
        ], capacities=4)
        is_valid, errors = bonus_board.validate()
        self.assertTrue(is_valid, errors)
        self.assertEqual(bonus_board.num_tubes, 7)
        self.assertEqual(bonus_board.total_balls, 20)
        self.assertEqual(bonus_board.mystery_ball_count, 15)
        self.assertEqual(bonus_board.known_ball_count, 5)
        self.assertEqual(bonus_board.empty_tubes_count, 2)
        self.assertTrue(bonus_board.has_mystery_balls)
        self.assertFalse(bonus_board.is_solved)

        # 9. BLACK rejection if explicitly inserted into active balls
        invalid_black_tube = TubeState(id=1, capacity=4, balls=["PINK", "BLACK"])
        b_bad = Board(tubes=[invalid_black_tube])
        is_valid, errors = b_bad.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("BLACK/EMPTY" in e for e in errors))


class TestMixedAndVariableCapacities(unittest.TestCase):
    """
    Unit tests for variable-capacity tubes (capacity 4, capacity 5, mixed 5/4 boards).
    """
    def test_capacity_4_matrix(self):
        # 0/4
        t0 = TubeState(id=1, capacity=4, balls=[])
        self.assertTrue(t0.is_empty)
        self.assertFalse(t0.is_full)
        self.assertEqual(t0.available_space, 4)
        self.assertEqual(t0.ball_count, 0)
        self.assertTrue(t0.is_solved)

        # 1/4
        t1 = TubeState(id=2, capacity=4, balls=["RED"])
        self.assertFalse(t1.is_empty)
        self.assertFalse(t1.is_full)
        self.assertEqual(t1.available_space, 3)
        self.assertEqual(t1.ball_count, 1)

        # 2/4
        t2 = TubeState(id=3, capacity=4, balls=["RED", "RED"])
        self.assertEqual(t2.available_space, 2)
        self.assertEqual(t2.ball_count, 2)

        # 3/4
        t3 = TubeState(id=4, capacity=4, balls=["RED", "RED", "RED"])
        self.assertEqual(t3.available_space, 1)
        self.assertEqual(t3.ball_count, 3)

        # 4/4
        t4 = TubeState(id=5, capacity=4, balls=["RED", "RED", "RED", "RED"])
        self.assertFalse(t4.is_empty)
        self.assertTrue(t4.is_full)
        self.assertEqual(t4.available_space, 0)
        self.assertEqual(t4.ball_count, 4)
        self.assertTrue(t4.is_solved)

    def test_capacity_5_matrix(self):
        # 0/5
        t0 = TubeState(id=1, capacity=5, balls=[])
        self.assertTrue(t0.is_empty)
        self.assertFalse(t0.is_full)
        self.assertEqual(t0.available_space, 5)
        self.assertEqual(t0.ball_count, 0)
        self.assertTrue(t0.is_solved)

        # 1/5
        t1 = TubeState(id=2, capacity=5, balls=["YELLOW"])
        self.assertFalse(t1.is_empty)
        self.assertFalse(t1.is_full)
        self.assertEqual(t1.available_space, 4)
        self.assertEqual(t1.ball_count, 1)

        # 2/5
        t2 = TubeState(id=3, capacity=5, balls=["YELLOW", "YELLOW"])
        self.assertEqual(t2.available_space, 3)
        self.assertEqual(t2.ball_count, 2)

        # 3/5
        t3 = TubeState(id=4, capacity=5, balls=["YELLOW", "YELLOW", "YELLOW"])
        self.assertEqual(t3.available_space, 2)
        self.assertEqual(t3.ball_count, 3)

        # 4/5
        t4 = TubeState(id=5, capacity=5, balls=["YELLOW", "YELLOW", "YELLOW", "YELLOW"])
        self.assertFalse(t4.is_full)
        self.assertEqual(t4.available_space, 1)
        self.assertEqual(t4.ball_count, 4)
        self.assertFalse(t4.is_solved)

        # 5/5
        t5 = TubeState(id=6, capacity=5, balls=["YELLOW", "YELLOW", "YELLOW", "YELLOW", "YELLOW"])
        self.assertTrue(t5.is_full)
        self.assertEqual(t5.available_space, 0)
        self.assertEqual(t5.ball_count, 5)
        self.assertTrue(t5.is_pure)
        self.assertTrue(t5.is_solved)

    def test_exact_5_5_5_5_5_4_board_model(self):
        board = Board.from_lists([
            ['YELLOW', 'YELLOW', 'YELLOW', 'GREEN', 'LIGHT_BLUE'],
            ['DARK_PURPLE', 'LIGHT_BLUE', 'DARK_PURPLE', 'GREEN', 'YELLOW'],
            ['GREEN', 'LIGHT_BLUE', 'DARK_PURPLE', 'YELLOW', 'DARK_PURPLE'],
            ['GREEN', 'LIGHT_BLUE', 'DARK_PURPLE', 'LIGHT_BLUE', 'GREEN'],
            [],
            []
        ], capacities=[5, 5, 5, 5, 5, 4])

        is_valid, errors = board.validate()
        self.assertTrue(is_valid, errors)
        self.assertEqual(board.num_tubes, 6)
        self.assertEqual(board.total_balls, 20)
        self.assertEqual(board.color_counts, {'YELLOW': 5, 'GREEN': 5, 'LIGHT_BLUE': 5, 'DARK_PURPLE': 5})
        self.assertEqual([t.capacity for t in board.tubes], [5, 5, 5, 5, 5, 4])
        self.assertEqual([t.ball_count for t in board.tubes], [5, 5, 5, 5, 0, 0])
        self.assertEqual([t.available_space for t in board.tubes], [0, 0, 0, 0, 5, 4])


if __name__ == "__main__":
    unittest.main()
