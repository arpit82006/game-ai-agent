"""
File: bonus/detector.py
Purpose:
    Reliable detection of game mode (NORMAL LEVEL vs BONUS / MYSTERY LEVEL).
"""

from __future__ import annotations
from models.board import Board


def is_bonus_level(board: Board | None) -> bool:
    """
    Determine if the current game state represents a SPECIAL / BONUS LEVEL.

    A level is identified as a bonus level if and only if unrevealed GRAY
    mystery balls are present on the board.

    Args:
        board (Board | None): The board state from the vision pipeline.

    Returns:
        bool: True if GRAY mystery balls are detected.
    """
    if board is None:
        return False
    return bool(board.has_mystery_balls)
