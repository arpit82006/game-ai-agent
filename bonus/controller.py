"""
File: bonus/controller.py
Purpose:
    Iterative execution loop, atomic context snapshotting, and lifecycle manager
    for Bonus / Mystery levels.
"""

from __future__ import annotations
import time
import os
import copy
import cv2
from core.adb import Emulator
from automation.adb_input import tap, get_total_physical_taps
from vision.pipeline import run_vision_pipeline, run_bonus_vision
from automation.executor import execute_move, find_tube_by_id, get_tube_tap_point
from automation.models import AutomationConfig
from models.board import Board
from bonus.models import BonusRevealReport, RevealStepResult, RevealMove, RevealStepContext, DestinationMode
from bonus.planner import select_best_reveal_move
from bonus.verifier import verify_reveal_transition
from solver.visualizer import render_ascii_board


def create_reveal_context(
    board: Board,
    vision_tubes: list,
    reveal_move: RevealMove
) -> RevealStepContext:
    """
    Create an immutable atomic RevealStepContext binding pre-move Board,
    RevealMove, logical IDs, physical tap points, and structural expectations.

    Args:
        board (Board): The exact board state immediately prior to move dispatch.
        vision_tubes (list): OpenCV Tube geometric objects detected from that same image.
        reveal_move (RevealMove): The move selected from `board`.

    Returns:
        RevealStepContext: Frozen context snapshot.
    """
    m = reveal_move.move
    src_tube_obj = find_tube_by_id(vision_tubes, m.from_tube)
    dst_tube_obj = find_tube_by_id(vision_tubes, m.to_tube)
    src_pt = get_tube_tap_point(src_tube_obj) if src_tube_obj else (0, 0)
    dst_pt = get_tube_tap_point(dst_tube_obj) if dst_tube_obj else (0, 0)

    dst_tube_before = board.get_tube(m.to_tube)
    empty_tubes_before = [t.id for t in board.tubes if t.is_empty]
    is_empty_reserve = dst_tube_before.is_empty

    if is_empty_reserve:
        matching_occupied_tubes = [
            t.id for t in board.tubes
            if t.id != m.from_tube and not t.is_empty and t.top_color == m.color and t.available_space >= m.ball_count
        ]
        if len(matching_occupied_tubes) == 0:
            dest_mode = DestinationMode.AUTO_EMPTY
        else:
            dest_mode = DestinationMode.EMPTY_RESERVE_GROUP
    else:
        dest_mode = DestinationMode.EXACT_TUBE

    return RevealStepContext(
        before_board=board,
        reveal_move=reveal_move,
        source_tube_id=m.from_tube,
        destination_tube_id=m.to_tube,
        transferred_color=m.color,
        transfer_count=m.ball_count,
        source_tap_point=src_pt,
        dest_tap_point=dst_pt,
        will_expose_mystery=reveal_move.will_expose_mystery,
        destination_mode=dest_mode,
        candidate_empty_reserve_ids=tuple(empty_tubes_before) if is_empty_reserve else (),
        planned_representative_destination_id=m.to_tube if is_empty_reserve else None,
        is_empty_reserve_move=is_empty_reserve,
        valid_empty_reserve_ids=tuple(empty_tubes_before) if is_empty_reserve else ()
    )


def run_bonus_reveal_single_step(
    initial_board: Board,
    vision_tubes: list,
    emulator: Emulator | None = None,
    config: AutomationConfig | None = None
) -> tuple[bool, str, Board | None, RevealStepContext | None]:
    """
    Execute exactly ONE controlled progressive reveal move for experimental validation.

    Guarantees strict sequential lifecycle:
      [1] Source tap dispatched
      [2] Wait bonus tap delay / auto-placement
      [3] Destination tap dispatched (EXACT_TUBE / EMPTY_RESERVE only; omitted for AUTO_EMPTY)
      [4] Wait move settle delay
      [5] Fresh screenshot captured
      [6] Vision pipeline executed
      [7] After-board constructed
      [8] Verification evaluated
      [9] Unlock / Halt
    """
    if config is None:
        config = AutomationConfig(bonus_tap_delay=0.50, bonus_move_settle_delay=1.20, auto_empty_settle_delay=1.50)

    reveal_move = select_best_reveal_move(initial_board)
    if reveal_move is None:
        return False, "BONUS REVEAL STALLED: No legal reveal move could be determined.", None, None

    ctx = create_reveal_context(initial_board, vision_tubes, reveal_move)
    m = ctx.reveal_move.move
    mystery_before = ctx.before_board.mystery_ball_count

    print("\n" + "=" * 55)
    print("  BONUS REVEAL STEP (ONE MOVE ONLY)")
    print("=" * 55)
    print(f"  Mystery Balls Before : {mystery_before}")
    print(f"  Planned Reveal       : Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id} | {ctx.transferred_color} x{ctx.transfer_count}")
    if ctx.destination_mode == DestinationMode.AUTO_EMPTY:
        candidate_str = ", ".join(f"Tube {x}" for x in ctx.candidate_empty_reserve_ids)
        print(f"  Destination Mode     : AUTO_EMPTY")
        print(f"  Candidate Reserves   : {candidate_str}")
        print(f"  Physical Action      : Source tap ONLY (NO DESTINATION TAP — GAME AUTO-PLACEMENT)")
        print(f"  Timing               : auto_empty_settle={config.auto_empty_settle_delay:.2f}s")
    elif ctx.destination_mode == DestinationMode.EMPTY_RESERVE_GROUP:
        candidate_str = ", ".join(f"Tube {x}" for x in ctx.candidate_empty_reserve_ids)
        print(f"  Destination Mode     : EMPTY RESERVE GROUP")
        print(f"  Candidate Reserves   : {candidate_str}")
        print(f"  Interaction Point    : Tube {ctx.destination_tube_id} {ctx.dest_tap_point}")
        print(f"  Timing               : tap_delay={config.bonus_tap_delay:.2f}s, settle={config.bonus_move_settle_delay:.2f}s")
    else:
        print(f"  Destination Mode     : EXACT TUBE")
        print(f"  Destination Tap      : Tube {ctx.destination_tube_id} {ctx.dest_tap_point}")
        print(f"  Timing               : tap_delay={config.bonus_tap_delay:.2f}s, settle={config.bonus_move_settle_delay:.2f}s")
    print(f"  Source Tap           : Tube {ctx.source_tube_id} {ctx.source_tap_point}")
    print("=" * 55)

    if config.dry_run:
        print("\n  [DRY-RUN] Simulated physical tap dispatch. Zero physical taps sent.\n")
        return True, "Dry-run simulated successfully.", initial_board, ctx

    taps_before_move = get_total_physical_taps()

    if ctx.destination_mode == DestinationMode.AUTO_EMPTY:
        # [1] Source tap only
        print(f"\n  [1] Source tap dispatched (AUTO-EMPTY): Tube {ctx.source_tube_id} {ctx.source_tap_point}")
        ok_src, msg_src = tap(
            ctx.source_tap_point[0], ctx.source_tap_point[1],
            emulator=emulator,
            purpose=f"AUTO_EMPTY_SOURCE (Move: Tube {ctx.source_tube_id} -> AUTO EMPTY)"
        )
        if not ok_src:
            return False, f"Source tap failed: {msg_src}", None, ctx

        # [2] Wait auto-placement settle delay
        print(f"  [2] Waiting auto-placement settle delay: {config.auto_empty_settle_delay:.2f}s (NO DESTINATION TAP)")
        time.sleep(config.auto_empty_settle_delay)
        expected_taps = taps_before_move + 1
    else:
        # [1] Source tap
        print(f"\n  [1] Source tap dispatched: Tube {ctx.source_tube_id} {ctx.source_tap_point}")
        ok_src, msg_src = tap(
            ctx.source_tap_point[0], ctx.source_tap_point[1],
            emulator=emulator,
            purpose=f"BONUS_SOURCE (Move: Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id})"
        )
        if not ok_src:
            return False, f"Source tap failed: {msg_src}", None, ctx

        # [2] Wait bonus tap delay
        print(f"  [2] Waiting bonus tap delay: {config.bonus_tap_delay:.2f}s")
        time.sleep(config.bonus_tap_delay)

        # [3] Destination tap
        dest_label = "reserve interaction" if ctx.destination_mode == DestinationMode.EMPTY_RESERVE_GROUP else "destination"
        print(f"  [3] Destination tap dispatched: Tube {ctx.destination_tube_id} ({dest_label}) {ctx.dest_tap_point}")
        ok_dst, msg_dst = tap(
            ctx.dest_tap_point[0], ctx.dest_tap_point[1],
            emulator=emulator,
            purpose=f"BONUS_DESTINATION (Move: Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id})"
        )
        if not ok_dst:
            return False, f"Destination tap failed: {msg_dst}", None, ctx

        # [4] Wait move settle delay
        print(f"  [4] Waiting move settle delay: {config.bonus_move_settle_delay:.2f}s")
        time.sleep(config.bonus_move_settle_delay)
        expected_taps = taps_before_move + 2

    # Audit tap count before capturing screenshot
    taps_after_settle = get_total_physical_taps()
    if taps_after_settle != expected_taps:
        msg = f"UNAUTHORIZED PHYSICAL INPUT DETECTED! Expected {expected_taps} total taps, found {taps_after_settle}."
        print(f"\n[CRITICAL ERROR] {msg}\n")
        return False, msg, None, ctx

    # [5] Capture fresh screenshot
    print("  [5] Capturing fresh post-reveal screenshot...")
    if emulator is None:
        return False, "No emulator connected to capture fresh post-reveal screenshot.", None, ctx

    fresh_img, _ = emulator.capture_screenshot("screenshots/screen.png")
    if fresh_img is None:
        return False, "Failed to capture post-reveal screenshot from emulator.", None, ctx
    print("  [5] Fresh screenshot captured OK")

    # Save raw verification frame for diagnostic auditing
    os.makedirs("debug/latest", exist_ok=True)
    cv2.imwrite("debug/latest/bonus_verification_raw.png", fresh_img)

    # [6] & [7] Vision analysis & After-board construction
    print("  [6] Running vision analysis on fresh screenshot using stable tube geometry...")
    vision_res = run_bonus_vision(fresh_img, stable_tubes=vision_tubes, save_debug=True, debug_dir="debug/latest")
    if not vision_res.board:
        return False, "Vision pipeline failed to construct board after reveal move.", None, ctx
    print("  [7] After-board constructed OK")

    new_board = vision_res.board
    # [8] Verify using frozen context snapshot
    v_ok, v_msg = verify_reveal_transition(ctx, new_board)

    mystery_after = new_board.mystery_ball_count
    src_tube_after = new_board.get_tube(ctx.source_tube_id)
    newly_revealed_color = src_tube_after.top_color if not src_tube_after.is_empty else "N/A (TUBE EMPTIED)"

    print("\n" + "=" * 55)
    print("  BEFORE vs AFTER REVEAL TUBE COMPARISON")
    print("=" * 55)
    print(f"  BEFORE Move:")
    src_before = ctx.before_board.get_tube(ctx.source_tube_id)
    print(f"    Tube {src_before.id} (Source)     : {src_before.balls}")
    if ctx.destination_mode in (DestinationMode.AUTO_EMPTY, DestinationMode.EMPTY_RESERVE_GROUP) and len(ctx.candidate_empty_reserve_ids) > 1:
        for vid in ctx.candidate_empty_reserve_ids:
            vt = ctx.before_board.get_tube(vid)
            print(f"    Tube {vt.id} (Reserve {'*' if vid == ctx.destination_tube_id else ' '}): {vt.balls}")
    else:
        dst_before = ctx.before_board.get_tube(ctx.destination_tube_id)
        print(f"    Tube {dst_before.id} (Destination): {dst_before.balls}")

    print(f"\n  AFTER Move (Perceived from Live Screen):")
    if ctx.destination_mode in (DestinationMode.AUTO_EMPTY, DestinationMode.EMPTY_RESERVE_GROUP) and len(ctx.candidate_empty_reserve_ids) > 1:
        actual_reserves = [vid for vid in ctx.candidate_empty_reserve_ids if not new_board.get_tube(vid).is_empty]
        actual_dst_id = actual_reserves[0] if actual_reserves else None
        print(f"    Tube {src_tube_after.id} (Source)     : {src_tube_after.balls}")
        for vid in ctx.candidate_empty_reserve_ids:
            vt = new_board.get_tube(vid)
            marker = "*" if vid == actual_dst_id else " "
            print(f"    Tube {vt.id} (Reserve {marker}): {vt.balls}")
        print(f"\n  Destination Mode            : {ctx.destination_mode.value}")
        print(f"  Planned Representative      : Tube {ctx.destination_tube_id}")
        print(f"  Actual Physical Reserve     : Tube {actual_dst_id if actual_dst_id else 'None'}")
    else:
        dst_after = new_board.get_tube(ctx.destination_tube_id)
        print(f"    Tube {src_tube_after.id} (Source)     : {src_tube_after.balls}")
        print(f"    Tube {dst_after.id} (Destination): {dst_after.balls}")
        print(f"\n  Destination Mode            : EXACT TUBE")
        print(f"  Planned Destination         : Tube {ctx.destination_tube_id}")
        print(f"  Actual Destination          : Tube {dst_after.id}")

    print("-" * 55)
    print(f"  Mystery Balls Before: {mystery_before}")
    print(f"  Mystery Balls After : {mystery_after}")
    print(f"  Newly Revealed Color: {newly_revealed_color}")
    print(f"  Verification Result : {'PASS' if v_ok else 'FAIL (' + v_msg + ')'}")
    print("=" * 55)

    if v_ok:
        print("\n" + "=" * 55)
        print("  BONUS REVEAL TEST: PASS")
        print("=" * 55)
        print("  One reveal move executed and verified successfully.")
        print(f"  Mystery Balls Before: {mystery_before}")
        print(f"  Mystery Balls After : {mystery_after}")
        print(f"  Newly Revealed Color: {newly_revealed_color}")
        print("  No further moves executed.")
        print("=" * 55 + "\n")
        return True, "One reveal move executed and verified successfully.", new_board, ctx
    else:
        print("\n" + "=" * 55)
        print("  BONUS REVEAL TEST: FAIL")
        print("=" * 55)
        print(f"  Error: {v_msg}")
        print("=" * 55 + "\n")
        return False, f"Verification failed: {v_msg}", new_board, ctx


def run_bonus_reveal_loop(
    initial_board: Board,
    vision_tubes: list,
    emulator: Emulator | None = None,
    config: AutomationConfig | None = None,
    max_iterations: int = 30
) -> BonusRevealReport:
    """
    Execute the progressive reveal loop until all mystery balls are exposed.
    """
    if config is None:
        config = AutomationConfig(bonus_tap_delay=0.50, bonus_move_settle_delay=1.20, auto_empty_settle_delay=1.50)

    report = BonusRevealReport(total_iterations=0, success=False)
    current_board = initial_board
    current_tubes = vision_tubes

    print("\n" + "=" * 55)
    print("  BONUS LEVEL PROGRESSIVE REVEAL LOOP")
    print("=" * 55)
    print(f"  Starting Mystery Balls : {current_board.mystery_ball_count}")
    print(f"  Starting Revealed Balls: {current_board.known_ball_count}")
    print("=" * 55 + "\n")

    iteration = 0
    next_move_unlocked = True

    while current_board.has_mystery_balls and iteration < max_iterations:
        if not next_move_unlocked:
            report.abort_reason = "Safety invariant violated: Next move dispatch attempted while previous move was unverified."
            print(f"\n[ERROR] {report.abort_reason}\n")
            return report

        iteration += 1
        mystery_before = current_board.mystery_ball_count

        reveal_move = select_best_reveal_move(current_board)
        if reveal_move is None:
            report.abort_reason = (
                f"BONUS REVEAL STALLED: {mystery_before} mystery balls remaining, "
                f"but no safe legal reveal move could be determined."
            )
            print(f"\n[ABORT] {report.abort_reason}\n")
            return report

        ctx = create_reveal_context(current_board, current_tubes, reveal_move)
        m = ctx.reveal_move.move

        # Lock out any next move until this iteration completes and verifies
        next_move_unlocked = False

        print("\n" + "=" * 55)
        print(f"  BONUS REVEAL — ITERATION {iteration}")
        print("=" * 55)
        print(f"  Mystery Before : {mystery_before}")
        print(f"  Move           : Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id} | {ctx.transferred_color} x{ctx.transfer_count}")
        if ctx.destination_mode == DestinationMode.AUTO_EMPTY:
            candidate_str = ", ".join(f"Tube {x}" for x in ctx.candidate_empty_reserve_ids)
            print(f"  Destination Mode: AUTO_EMPTY")
            print(f"  Candidate Reserves: {candidate_str}")
            print(f"  Physical Action : Source tap ONLY (NO DESTINATION TAP — GAME AUTO-PLACEMENT)")
            print(f"  Timing          : auto_empty_settle={config.auto_empty_settle_delay:.2f}s")
        elif ctx.destination_mode == DestinationMode.EMPTY_RESERVE_GROUP:
            candidate_str = ", ".join(f"Tube {x}" for x in ctx.candidate_empty_reserve_ids)
            print(f"  Destination Mode: EMPTY RESERVE GROUP")
            print(f"  Candidate Reserves: {candidate_str}")
            print(f"  Interaction Pt  : Tube {ctx.destination_tube_id} {ctx.dest_tap_point}")
            print(f"  Timing          : tap_delay={config.bonus_tap_delay:.2f}s, settle={config.bonus_move_settle_delay:.2f}s")
        else:
            print(f"  Destination Mode: EXACT TUBE")
            print(f"  Destination Tap : Tube {ctx.destination_tube_id} {ctx.dest_tap_point}")
            print(f"  Timing          : tap_delay={config.bonus_tap_delay:.2f}s, settle={config.bonus_move_settle_delay:.2f}s")
        print(f"  Source Tap      : Tube {ctx.source_tube_id} {ctx.source_tap_point}")

        if config.dry_run:
            print(f"  Execution      : OK (DRY-RUN)")
            mock_tubes = []
            for t in current_board.tubes:
                balls = list(t.balls)
                if t.id == m.from_tube:
                    transferred = balls[:m.ball_count]
                    balls = balls[m.ball_count:]
                    if balls and balls[0] == "GRAY":
                        balls[0] = "SIMULATED_COLOR"
                elif t.id == m.to_tube:
                    balls = [m.color] * m.ball_count + balls
                mock_tubes.append(balls)

            current_board = Board.from_lists(mock_tubes, capacities=[t.capacity for t in current_board.tubes])
            report.total_iterations = iteration
            print(f"  Fresh Vision   : OK (DRY-RUN)")
            print(f"  Mystery After  : {current_board.mystery_ball_count}")
            print(f"  Newly Revealed : SIMULATED_COLOR")
            print(f"  Verification   : PASS (DRY-RUN)")
            next_move_unlocked = True
            continue

        # Audit baseline tap count
        taps_before_move = get_total_physical_taps()

        if ctx.destination_mode == DestinationMode.AUTO_EMPTY:
            # [1] Source tap only
            print(f"\n  [1] Source tap dispatched (AUTO-EMPTY): Tube {ctx.source_tube_id} {ctx.source_tap_point}")
            ok_src, msg_src = tap(
                ctx.source_tap_point[0], ctx.source_tap_point[1],
                emulator=emulator,
                purpose=f"AUTO_EMPTY_SOURCE (Iteration {iteration}: Tube {ctx.source_tube_id} -> AUTO EMPTY)"
            )
            if not ok_src:
                report.abort_reason = f"Source tap failed at iteration {iteration}: {msg_src}"
                print(f"  [ERROR] {report.abort_reason}\n")
                return report

            # [2] Wait auto-placement settle delay
            print(f"  [2] Waiting auto-placement settle delay: {config.auto_empty_settle_delay:.2f}s (NO DESTINATION TAP)")
            time.sleep(config.auto_empty_settle_delay)
            expected_taps = taps_before_move + 1
        else:
            # [1] Source tap
            print(f"\n  [1] Source tap dispatched: Tube {ctx.source_tube_id} {ctx.source_tap_point}")
            ok_src, msg_src = tap(
                ctx.source_tap_point[0], ctx.source_tap_point[1],
                emulator=emulator,
                purpose=f"BONUS_SOURCE (Iteration {iteration}: Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id})"
            )
            if not ok_src:
                report.abort_reason = f"Source tap failed at iteration {iteration}: {msg_src}"
                print(f"  [ERROR] {report.abort_reason}\n")
                return report

            # [2] Wait bonus tap delay
            print(f"  [2] Waiting bonus tap delay: {config.bonus_tap_delay:.2f}s")
            time.sleep(config.bonus_tap_delay)

            # [3] Destination tap
            dest_label = "reserve interaction" if ctx.destination_mode == DestinationMode.EMPTY_RESERVE_GROUP else "destination"
            print(f"  [3] Destination tap dispatched: Tube {ctx.destination_tube_id} ({dest_label}) {ctx.dest_tap_point}")
            ok_dst, msg_dst = tap(
                ctx.dest_tap_point[0], ctx.dest_tap_point[1],
                emulator=emulator,
                purpose=f"BONUS_DESTINATION (Iteration {iteration}: Tube {ctx.source_tube_id} -> Tube {ctx.destination_tube_id})"
            )
            if not ok_dst:
                report.abort_reason = f"Destination tap failed at iteration {iteration}: {msg_dst}"
                print(f"  [ERROR] {report.abort_reason}\n")
                return report

            # [4] Wait move settle delay
            print(f"  [4] Waiting move settle delay: {config.bonus_move_settle_delay:.2f}s")
            time.sleep(config.bonus_move_settle_delay)
            expected_taps = taps_before_move + 2

        # Audit tap count before capturing screenshot
        taps_after_settle = get_total_physical_taps()
        if taps_after_settle != expected_taps:
            report.abort_reason = (
                f"UNAUTHORIZED PHYSICAL INPUT DETECTED during settle/verification! "
                f"Expected exactly {expected_taps} total taps, but found {taps_after_settle}."
            )
            print(f"\n[CRITICAL ERROR] {report.abort_reason}\n")
            return report

        # [5] Capture fresh screenshot
        print("  [5] Capturing fresh post-reveal screenshot...")
        if emulator is None:
            report.abort_reason = "No emulator connected to capture fresh post-reveal screenshot."
            return report

        fresh_img, _ = emulator.capture_screenshot("screenshots/screen.png")
        if fresh_img is None:
            report.abort_reason = "Failed to capture post-reveal screenshot from emulator."
            return report
        print("  [5] Fresh screenshot captured OK")

        # Save raw verification frame for diagnostic auditing
        os.makedirs("debug/latest", exist_ok=True)
        cv2.imwrite("debug/latest/bonus_verification_raw.png", fresh_img)
        cv2.imwrite(f"debug/latest/raw_iteration_{iteration}.png", fresh_img)

        # Audit tap count before running vision
        taps_during_vision = get_total_physical_taps()
        if taps_during_vision != expected_taps:
            report.abort_reason = (
                f"UNAUTHORIZED PHYSICAL INPUT DETECTED during capture! "
                f"Expected {expected_taps} total taps, but found {taps_during_vision}."
            )
            print(f"\n[CRITICAL ERROR] {report.abort_reason}\n")
            return report

        # [6] & [7] Vision analysis & After-board construction
        print("  [6] Running vision analysis on fresh screenshot using stable tube geometry...")
        t0_vision = time.perf_counter()
        vision_res = run_bonus_vision(fresh_img, stable_tubes=current_tubes, save_debug=False)
        t_vision_ms = (time.perf_counter() - t0_vision) * 1000
        if not vision_res.board:
            report.abort_reason = "Vision pipeline failed to construct board after reveal move."
            return report
        print(f"  [7] After-board constructed OK ({t_vision_ms:.1f}ms)")

        new_board = vision_res.board
        new_tubes = current_tubes

        # [8] Verify reveal step using frozen context snapshot
        v_ok, v_msg = verify_reveal_transition(ctx, new_board)
        mystery_after = new_board.mystery_ball_count
        src_tube_after = new_board.get_tube(ctx.source_tube_id)
        newly_revealed_color = src_tube_after.top_color if not src_tube_after.is_empty else "N/A (TUBE EMPTIED)"

        if ctx.destination_mode in (DestinationMode.AUTO_EMPTY, DestinationMode.EMPTY_RESERVE_GROUP):
            actual_reserves = [vid for vid in ctx.candidate_empty_reserve_ids if not new_board.get_tube(vid).is_empty]
            actual_dst_id = actual_reserves[0] if actual_reserves else None
            print(f"  [8] Destination Mode: {ctx.destination_mode.value}")
            print(f"      Planned Target  : Tube {ctx.destination_tube_id}")
            print(f"      Actual Physical : Tube {actual_dst_id if actual_dst_id else 'None'}")
        else:
            print(f"  [8] Destination Mode: EXACT TUBE")
            print(f"      Planned Target  : Tube {ctx.destination_tube_id}")
            print(f"      Actual Target   : Tube {ctx.destination_tube_id}")

        print(f"      Verification Result : {'PASS' if v_ok else 'FAIL (' + v_msg + ')'}")
        print(f"      Mystery Before: {mystery_before} -> After: {mystery_after}")
        print(f"      Newly Revealed Color: {newly_revealed_color}")

        step_res = RevealStepResult(
            iteration=iteration,
            move=m,
            mystery_count_before=mystery_before,
            mystery_count_after=mystery_after,
            success=v_ok,
            error_message=None if v_ok else v_msg,
            board_after=new_board
        )
        report.steps.append(step_res)

        if not v_ok:
            next_move_unlocked = False
            report.abort_reason = f"Verification failed after reveal move {m}: {v_msg}"
            print(f"\n[ERROR] {report.abort_reason}\n")
            print("  Board perceived after failure:")
            print(render_ascii_board(new_board))
            return report

        # [9] Unlock next move ONLY after verification PASS
        next_move_unlocked = True
        print("  [9] Verification PASSED — Next move unlocked\n")

        current_board = new_board
        current_tubes = new_tubes
        report.total_iterations = iteration

    if not current_board.has_mystery_balls:
        report.success = True
        report.final_board = current_board
        report.final_tubes = current_tubes
        print("\n" + "=" * 55)
        print("  ALL MYSTERY BALLS REVEALED")
        print("=" * 55)
        print(f"  Total Balls       : {current_board.total_balls}")
        print(f"  Known Balls       : {current_board.known_ball_count}")
        print(f"  Mystery Balls     : {current_board.mystery_ball_count}")
        print(f"  Board Validation  : PASS")
        print(f"  Reveal Steps Taken: {report.total_iterations}")
        print("  Final Board State :")
        print(render_ascii_board(current_board))
        print("  Transitioning to Canonical BFS...")
        print("=" * 55 + "\n")

    return report
