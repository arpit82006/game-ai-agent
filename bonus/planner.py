"""
File: bonus/planner.py
Purpose:
    Safe reveal move planning and prioritization for bonus mystery levels.
"""

from __future__ import annotations
from models.board import Board
from solver.models import Move
from bonus.models import RevealMove


def get_legal_reveal_moves(board: Board) -> list[RevealMove]:
    """
    Generate all legal Ball Sort moves using currently revealed top balls,
    evaluating and scoring their capability to expose GRAY mystery balls.

    Move Rules:
      1. Source must have a known (non-GRAY) top ball.
      2. Destination must have available space.
      3. Destination must be empty OR match source top color (GRAY is never matching).
      4. Mystery balls underneath are NEVER moved directly.

    Prioritization:
      - Priority 20: Merging with matching top color in another tube AND exposes GRAY underneath.
      - Priority 15: Merging with matching top color without directly exposing GRAY (frees up space).
      - Priority 10: Moving top ball into an empty reserve tube AND exposes GRAY underneath.
      - Priority  1: Moving top ball into an empty reserve tube without exposing GRAY (last resort).

    Args:
        board (Board): The current partially revealed board.

    Returns:
        list[RevealMove]: List of legal reveal moves sorted descending by priority.
    """
    moves: list[RevealMove] = []
    first_empty_tube_id = next((t.id for t in board.tubes if t.is_empty), None)

    for src in board.tubes:
        if src.is_empty:
            continue
        src_top_color = src.top_color
        if not src_top_color or src_top_color == "GRAY":
            continue

        src_same_cnt = src.top_same_color_count
        # Check if removing this top group immediately exposes a mystery ball
        remaining_balls = src.balls[src_same_cnt:]
        exposes_gray = bool(remaining_balls and remaining_balls[0] == "GRAY")

        for dst in board.tubes:
            if dst.id == src.id:
                continue
            if dst.available_space == 0:
                continue

            transfer_count = min(src_same_cnt, dst.available_space)

            if dst.is_empty:
                # Symmetry pruning: only move to the first empty tube
                if dst.id != first_empty_tube_id:
                    continue
                # Do not move a pure known-color tube into an empty tube
                if src.is_pure:
                    continue

                priority = 10 if exposes_gray else 1
                reason = (
                    "Exposes mystery ball into empty reserve"
                    if exposes_gray else
                    "Transfers top ball to empty reserve"
                )

                moves.append(
                    RevealMove(
                        move=Move(
                            from_tube=src.id,
                            to_tube=dst.id,
                            ball_count=transfer_count,
                            color=src_top_color
                        ),
                        will_expose_mystery=exposes_gray,
                        priority=priority,
                        reason=reason
                    )
                )
            else:
                # Destination is not empty: must match color and not be GRAY
                if dst.top_color == src_top_color and dst.top_color != "GRAY":
                    priority = 20 if exposes_gray else 15
                    reason = (
                        "Merges with matching color and exposes mystery ball"
                        if exposes_gray else
                        "Merges with matching color"
                    )

                    moves.append(
                        RevealMove(
                            move=Move(
                                from_tube=src.id,
                                to_tube=dst.id,
                                ball_count=transfer_count,
                                color=src_top_color
                            ),
                            will_expose_mystery=exposes_gray,
                            priority=priority,
                            reason=reason
                        )
                    )

    # Sort descending by priority score
    moves.sort(key=lambda rm: rm.priority, reverse=True)
    return moves


def select_best_reveal_move(board: Board) -> RevealMove | None:
    """
    Select the highest-priority safe reveal move for the given board.

    Args:
        board (Board): The current board state.

    Returns:
        RevealMove | None: Best reveal move, or None if no safe reveal move exists.
    """
    legal_moves = get_legal_reveal_moves(board)
    return legal_moves[0] if legal_moves else None
