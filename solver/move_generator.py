"""
File: solver/move_generator.py

Purpose:
    Generates all legal and non-redundant moves from a given Board state.
"""

from __future__ import annotations
from models.board import Board, TubeState
from solver.models import Move


def get_valid_moves(board: Board) -> list[Move]:
    """
    Generate all valid and productive moves from the current Board state.

    Move Rules:
      1. Source tube must not be empty.
      2. Source tube must not already be completely solved (full and monochromatic).
      3. Destination tube must not be full.
      4. Source and Destination must be different tubes.
      5. Destination must be empty OR have the same top color as Source.
      6. Transferred quantity is min(source.top_same_color_count, destination.available_space).

    Safe Symmetry & Pruning Rules:
      - Moving a pure monochromatic tube into an empty tube is skipped (isomorphic no-op).
      - If multiple empty destination tubes exist, only the first empty tube is targeted
        (all empty tubes are structurally identical).

    Args:
        board (Board): Current puzzle board state.

    Returns:
        list[Move]: List of valid Move instances.
    """
    moves: list[Move] = []
    first_empty_tube_id: int | None = None

    # Identify the first empty tube for symmetry reduction
    for tube in board.tubes:
        if tube.is_empty:
            first_empty_tube_id = tube.id
            break

    for src in board.tubes:
        # Rule 1: Cannot move from empty tube
        if src.is_empty:
            continue

        # Rule 2: Do not move from a completely finished tube (full & pure)
        if src.is_solved:
            continue

        src_top_color = src.top_color
        src_top_count = src.top_same_color_count

        for dst in board.tubes:
            # Rule 4: Source and destination must be distinct
            if src.id == dst.id:
                continue

            # Rule 3: Destination cannot be full
            if dst.is_full:
                continue

            dst_space = dst.available_space
            transfer_count = min(src_top_count, dst_space)
            if transfer_count <= 0:
                continue

            if dst.is_empty:
                # Symmetry pruning: only move to the first empty tube
                if dst.id != first_empty_tube_id:
                    continue

                # Pruning: do not move a pure partial tube into an empty tube (creates identical state)
                if src.is_pure:
                    continue

                moves.append(
                    Move(
                        from_tube=src.id,
                        to_tube=dst.id,
                        ball_count=transfer_count,
                        color=src_top_color
                    )
                )
            else:
                # Rule 5: Non-empty destination must match source top color
                if dst.top_color == src_top_color:
                    moves.append(
                        Move(
                            from_tube=src.id,
                            to_tube=dst.id,
                            ball_count=transfer_count,
                            color=src_top_color
                        )
                    )

    return moves
