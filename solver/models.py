"""
File: solver/models.py

Purpose:
    Data structures for the Ball Sort puzzle solver.
    Completely decoupled from OpenCV, image coordinates, ADB, and emulator details.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Move:
    """
    Represents a single atomic move action transferring balls between two tubes.

    Attributes:
        from_tube (int): 1-based ID of the source tube.
        to_tube (int): 1-based ID of the destination tube.
        ball_count (int): Number of contiguous matching balls transferred.
        color (str): Color of the balls being moved.
    """
    from_tube: int
    to_tube: int
    ball_count: int
    color: str

    def __repr__(self) -> str:
        return f"Tube {self.from_tube} -> Tube {self.to_tube} | {self.color} x{self.ball_count}"

    def __str__(self) -> str:
        return f"Tube {self.from_tube} -> Tube {self.to_tube} | {self.color} x{self.ball_count}"


@dataclass
class SolverResult:
    """
    Outcome of a puzzle solving search.

    Attributes:
        solved (bool): True if a complete solution sequence was found.
        moves (list[Move]): Sequence of moves leading to the solved state.
        move_count (int): Total number of move actions in the solution.
        states_explored (int): Number of unique board states evaluated during search.
        elapsed_time (float): Search execution time in seconds.
        failure_reason (str | None): Descriptive failure message if not solved.
    """
    solved: bool
    moves: list[Move] = field(default_factory=list)
    move_count: int = 0
    states_explored: int = 0
    elapsed_time: float = 0.0
    failure_reason: str | None = None
