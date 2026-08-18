"""
File: bonus/verifier.py
Purpose:
    Structural and invariant verification of post-reveal states in bonus levels.
"""

from __future__ import annotations
from models.board import Board
from solver.models import Move
from bonus.models import RevealStepContext


def verify_reveal_transition(
    board_before_or_ctx: Board | RevealStepContext,
    board_after: Board,
    move: Move | None = None,
    expected_to_expose: bool = True
) -> tuple[bool, str]:
    """
    Perform structural and physical consistency verification after a reveal move.

    Accepts either a frozen RevealStepContext (guaranteeing exact pre-move snapshot alignment)
    or direct (board_before, board_after, move, expected_to_expose) arguments.

    Unlike normal levels where complete color states are known in advance,
    bonus levels only verify invariant conservation and structural legitimacy:
      1. Post-move board passes logical validation.
      2. Total physical ball count is strictly conserved.
      3. Total tube count and individual capacities are conserved.
      4. Destination tube has received the transferred ball on top.
      5. If expected_to_expose is True:
         - Source tube top ball is no longer GRAY (it has turned into a real color), OR
         - Source tube became empty, OR
         - Total mystery ball count decreased.

    Args:
        board_before_or_ctx (Board | RevealStepContext): State prior to the physical move or context snapshot.
        board_after (Board): State perceived from fresh post-move screenshot.
        move (Move | None): The move executed (optional if RevealStepContext is provided).
        expected_to_expose (bool): Whether the move was planned to expose a GRAY ball.

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    if isinstance(board_before_or_ctx, RevealStepContext):
        ctx = board_before_or_ctx
        board_before = ctx.before_board
        move = ctx.reveal_move.move
        expected_to_expose = ctx.will_expose_mystery
        is_empty_reserve = ctx.is_empty_reserve_move
        valid_empty_reserve_ids = ctx.valid_empty_reserve_ids
    else:
        board_before = board_before_or_ctx
        if move is None:
            raise ValueError("Move must be provided if not passing a RevealStepContext.")
        is_empty_reserve = board_before.get_tube(move.to_tube).is_empty
        empty_before = [t.id for t in board_before.tubes if t.is_empty]
        valid_empty_reserve_ids = tuple(empty_before) if is_empty_reserve else ()

    is_valid, errors = board_after.validate()
    if not is_valid:
        return False, f"Post-reveal board validation failed: {'; '.join(errors)}"

    if board_after.total_balls != board_before.total_balls:
        return False, f"Ball count violated: before={board_before.total_balls}, after={board_after.total_balls}"

    if board_after.num_tubes != board_before.num_tubes:
        return False, f"Tube count changed: before={board_before.num_tubes}, after={board_after.num_tubes}"

    for t_before, t_after in zip(board_before.tubes, board_after.tubes):
        if t_after.capacity != t_before.capacity:
            return False, f"Tube {t_after.id} capacity changed from {t_before.capacity} to {t_after.capacity}"

    # Destination verification
    if is_empty_reserve and len(valid_empty_reserve_ids) > 1:
        # Multiple empty reserves were present before the move.
        # Check that exactly ONE of them received the transferred ball and has the exact color/count.
        receiving_tubes = [
            tid for tid in valid_empty_reserve_ids
            if not board_after.get_tube(tid).is_empty
        ]
        if len(receiving_tubes) == 0:
            return False, f"Transferred ball disappeared: none of empty reserve tubes {valid_empty_reserve_ids} received the ball"
        if len(receiving_tubes) > 1:
            return False, f"Ball duplication / invalid state: multiple empty reserve tubes {receiving_tubes} became occupied simultaneously"

        actual_dst_id = receiving_tubes[0]
        actual_dst_tube = board_after.get_tube(actual_dst_id)
        if actual_dst_tube.top_color != move.color or actual_dst_tube.ball_count != move.ball_count:
            return False, f"Empty reserve Tube {actual_dst_id} received {actual_dst_tube.top_color} x{actual_dst_tube.ball_count}, expected {move.color} x{move.ball_count}"

        # Verify all OTHER empty reserves in the group remained completely empty
        for oid in valid_empty_reserve_ids:
            if oid != actual_dst_id and not board_after.get_tube(oid).is_empty:
                return False, f"Empty reserve Tube {oid} unexpectedly became non-empty"
    else:
        # Explicit/occupied destination or single empty tube: exact tube match required
        dst_tube = board_after.get_tube(move.to_tube)
        dst_before = board_before.get_tube(move.to_tube)
        expected_dst_count = dst_before.ball_count + move.ball_count
        if dst_tube.top_color != move.color:
            return False, f"Destination Tube {move.to_tube} top color is {dst_tube.top_color}, expected {move.color}"
        if dst_tube.ball_count != expected_dst_count:
            return False, f"Destination Tube {move.to_tube} ball count is {dst_tube.ball_count}, expected {expected_dst_count}"

    src_tube = board_after.get_tube(move.from_tube)
    if expected_to_expose and not src_tube.is_empty:
        if src_tube.top_color == "GRAY":
            return False, f"Source Tube {move.from_tube} top ball is still perceived as GRAY (expected revealed color)"

    if expected_to_expose:
        if board_after.mystery_ball_count >= board_before.mystery_ball_count:
            return False, (
                f"Mystery ball count did not decrease: before={board_before.mystery_ball_count}, "
                f"after={board_after.mystery_ball_count}"
            )

    return True, "Reveal step structurally verified."
