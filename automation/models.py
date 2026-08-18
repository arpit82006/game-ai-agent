"""
File: automation/models.py

Purpose:
    Configuration and result models for Ball Sort move execution and automation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from models.board import Board
from solver.models import Move


@dataclass
class AutomationConfig:
    """
    Configuration parameters for ADB move execution and post-move verification.

    Attributes:
        tap_delay (float): Seconds to wait between tapping source and destination tube.
        move_settle_delay (float): Seconds to wait after the destination tap for in-game animation to complete.
        verification_delay (float): Seconds to wait before capturing a post-move verification screenshot.
        dry_run (bool): If True, computes and logs taps without sending physical ADB input.
    """
    tap_delay: float = 0.35
    move_settle_delay: float = 2.80
    verification_delay: float = 0.40
    dry_run: bool = False


@dataclass
class StepResult:
    """
    Outcome of executing a single move in the game.

    Attributes:
        step_number (int): Move sequence index (1-based).
        move (Move): The solver Move executed.
        source_tap (tuple[int, int]): Screen coordinates (X, Y) tapped for the source tube.
        dest_tap (tuple[int, int]): Screen coordinates (X, Y) tapped for the destination tube.
        expected_board (Board): The simulated expected board state after this move.
        actual_board (Board | None): The board state perceived by vision after this move.
        verified (bool): True if actual_board matches expected_board.
        error_message (str | None): Detail of mismatch or execution failure, if any.
    """
    step_number: int
    move: Move
    source_tap: tuple[int, int]
    dest_tap: tuple[int, int]
    expected_board: Board
    actual_board: Board | None = None
    verified: bool = False
    error_message: str | None = None


@dataclass
class ExecutionReport:
    """
    Summary report of an automation session.

    Attributes:
        total_moves_planned (int): Total moves in the solver solution.
        moves_executed (int): Number of moves physically executed.
        success (bool): True if all executed moves succeeded and verified.
        steps (list[StepResult]): List of individual step execution outcomes.
        abort_reason (str | None): Failure or abort message if session stopped early.
    """
    total_moves_planned: int
    moves_executed: int = 0
    success: bool = False
    steps: list[StepResult] = field(default_factory=list)
    abort_reason: str | None = None
