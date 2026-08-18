"""
File: bonus/models.py
Purpose:
    Data structures, context snapshots, and report classes for progressive mystery ball revealing.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from models.board import Board
from solver.models import Move


class RevealLifecycleState(Enum):
    """Execution lifecycle state machine for a single reveal iteration."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    FROZEN = "FROZEN"
    SOURCE_TAPPED = "SOURCE_TAPPED"
    DESTINATION_TAPPED = "DESTINATION_TAPPED"
    SETTLING = "SETTLING"
    CAPTURING = "CAPTURING"
    ANALYZING = "ANALYZING"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    UNLOCKED_FOR_NEXT_MOVE = "UNLOCKED_FOR_NEXT_MOVE"
    HALTED = "HALTED"


class DestinationMode(Enum):
    """Destination interaction classification for progressive reveal moves."""
    EXACT_TUBE = "EXACT_TUBE"
    EMPTY_RESERVE_GROUP = "EMPTY_RESERVE_GROUP"
    AUTO_EMPTY = "AUTO_EMPTY"


@dataclass
class RevealMove:
    """
    A planned physical reveal move aimed at uncovering unrevealed mystery balls.

    Attributes:
        move (Move): Standard Move object with physical source and destination tube IDs.
        will_expose_mystery (bool): True if executing this move will uncover a GRAY ball beneath.
        priority (int): Heuristic priority score (higher is executed first).
        reason (str): Human-readable explanation of why this reveal action was chosen.
    """
    move: Move
    will_expose_mystery: bool
    priority: int
    reason: str


@dataclass(frozen=True)
class RevealStepContext:
    """
    Immutable atomic snapshot for a single reveal iteration.

    Binds the pre-move board state, the planned RevealMove, physical tube IDs,
    tap coordinates, and structural expectations together so that planning,
    execution, and verification are guaranteed to operate on the exact same state.

    Attributes:
        before_board (Board): The exact Board state perceived immediately before move execution.
        reveal_move (RevealMove): The frozen RevealMove selected from before_board.
        source_tube_id (int): 1-based logical tube ID of the source tube.
        destination_tube_id (int): 1-based logical tube ID of the destination tube.
        transferred_color (str): Color name being transferred.
        transfer_count (int): Number of balls being transferred.
        source_tap_point (tuple[int, int]): Screen coordinates (X, Y) of source tap.
        dest_tap_point (tuple[int, int]): Screen coordinates (X, Y) of destination tap.
        will_expose_mystery (bool): True if the move is expected to uncover a GRAY ball.
        destination_mode (DestinationMode): EXACT_TUBE for occupied tubes vs EMPTY_RESERVE_GROUP.
        candidate_empty_reserve_ids (tuple[int, ...]): All empty reserve tube IDs prior to move.
        planned_representative_destination_id (int): Logical representative empty tube ID.
        is_empty_reserve_move (bool): True if destination was in the empty reserve group.
        valid_empty_reserve_ids (tuple[int, ...]): Candidate reserve tube IDs.
    """
    before_board: Board
    reveal_move: RevealMove
    source_tube_id: int
    destination_tube_id: int
    transferred_color: str
    transfer_count: int
    source_tap_point: tuple[int, int]
    dest_tap_point: tuple[int, int]
    will_expose_mystery: bool
    destination_mode: DestinationMode = DestinationMode.EXACT_TUBE
    candidate_empty_reserve_ids: tuple[int, ...] = ()
    planned_representative_destination_id: int | None = None
    is_empty_reserve_move: bool = False
    valid_empty_reserve_ids: tuple[int, ...] = ()


@dataclass
class RevealStepResult:
    """
    Outcome of a single progressive reveal step.

    Attributes:
        iteration (int): 1-based loop counter.
        move (Move): Move executed.
        mystery_count_before (int): Number of unrevealed balls before this move.
        mystery_count_after (int): Number of unrevealed balls perceived after this move.
        success (bool): True if structural verification passed.
        error_message (str | None): Verification failure explanation if failed.
        board_after (Board | None): Newly perceived board state after the move.
    """
    iteration: int
    move: Move
    mystery_count_before: int
    mystery_count_after: int
    success: bool
    error_message: str | None = None
    board_after: Board | None = None


@dataclass
class BonusRevealReport:
    """
    Complete summary of a progressive reveal session.

    Attributes:
        total_iterations (int): Total reveal actions executed.
        success (bool): True if all mystery balls were revealed and verification succeeded.
        final_board (Board | None): Fully revealed Board state ready for normal solver.
        steps (list[RevealStepResult]): Step-by-step history of reveal iterations.
        abort_reason (str | None): Failure or stall reason if aborted early.
    """
    total_iterations: int = 0
    success: bool = False
    final_board: Board | None = None
    final_tubes: list = field(default_factory=list)
    steps: list[RevealStepResult] = field(default_factory=list)
    abort_reason: str | None = None
