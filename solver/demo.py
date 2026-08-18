"""
File: solver/demo.py

Purpose:
    Standalone, offline demo for the Ball Sort puzzle solver.
    Runs purely in memory without requiring BlueStacks, ADB, OpenCV, or screenshots.

Usage:
    python -m solver.demo
    or
    python solver/demo.py
"""

from __future__ import annotations
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board import Board
from solver import solve, replay_and_visualize_solution, render_ascii_board


def run_demo() -> int:
    print("\n" + "=" * 65)
    print("  BALL SORT AI — STANDALONE SOLVER DEMO")
    print("  (Offline In-Memory Puzzle Solving & Step-by-Step Replay)")
    print("=" * 65)

    # ── Define Sample Puzzle
    # 4 colors (RED, BLUE, GREEN, YELLOW), 6 tubes (4 occupied + 2 empty reserve tubes)
    sample_puzzle = [
        ["RED", "BLUE", "GREEN", "RED"],        # Tube 1 (4/4)
        ["BLUE", "YELLOW", "RED", "YELLOW"],    # Tube 2 (4/4)
        ["GREEN", "GREEN", "BLUE", "YELLOW"],   # Tube 3 (4/4)
        ["YELLOW", "RED", "BLUE", "GREEN"],     # Tube 4 (4/4)
        [],                                     # Tube 5 (0/4 empty)
        []                                      # Tube 6 (0/4 empty)
    ]

    board = Board.from_lists(sample_puzzle, capacities=4)

    print("\n[Step 1] Initial Board Configuration:")
    print(render_ascii_board(board, title="Starting State"))

    is_valid, errors = board.validate()
    if not is_valid:
        print(f"\n[ERROR] Board validation failed: {'; '.join(errors)}")
        return 1

    print("\n[Step 2] Executing Breadth-First Search (BFS)...")
    result = solve(board, verbose=True, progress_interval=200)

    if not result.solved:
        print(f"\n[ERROR] Puzzle could not be solved: {result.failure_reason}")
        return 1

    print(f"[Step 3] Optimal Solution Found ({result.move_count} moves in {result.elapsed_time:.4f}s).")
    print(f"         Total States Explored: {result.states_explored:,}\n")

    print("[Step 4] Replaying Solution Move-by-Move in Memory:")
    success, msg, history = replay_and_visualize_solution(board, result.moves, print_steps=True)

    if not success:
        print(f"\n[ERROR] Replay verification failed: {msg}")
        return 1

    print("=" * 65)
    print("  DEMO COMPLETED SUCCESSFULLY — 100% VERIFIED")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
