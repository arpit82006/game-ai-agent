"""
File: automation/executor.py

Purpose:
    Translates abstract solver Move objects into physical screen taps on detected tubes.
"""

from __future__ import annotations
import time
from models.tube import Tube
from models.board import Board
from solver.models import Move
from core.adb import Emulator
from automation.models import AutomationConfig, ExecutionReport
from automation.adb_input import tap


def get_tube_tap_point(tube: Tube) -> tuple[int, int]:
    """
    Calculate a safe screen tap coordinate inside the body of a detected tube.

    Taps the horizontal center and vertical mid-body (55% from top), avoiding the
    top rim, UI elements above the tube, and bottom curved glass.

    Args:
        tube (Tube): OpenCV geometric Tube object.

    Returns:
        tuple[int, int]: (X, Y) pixel coordinates on the emulator display.
    """
    center_x = int(tube.x + tube.width / 2.0)
    body_y = int(tube.y + tube.height * 0.55)
    return center_x, body_y


def find_tube_by_id(tubes: list[Tube], tube_id: int) -> Tube | None:
    """Find a geometric Tube object by its 1-based ID."""
    for t in tubes:
        if t.id == tube_id:
            return t
    return None


def execute_move(
    move: Move,
    tubes: list[Tube],
    config: AutomationConfig | None = None,
    emulator: Emulator | None = None
) -> tuple[bool, str, tuple[int, int], tuple[int, int]]:
    """
    Execute a single Move action by tapping the source tube then the destination tube.

    Workflow:
      1. Map source tube ID to current screen coordinates.
      2. Map destination tube ID to current screen coordinates.
      3. Tap source tube.
      4. Wait tap_delay.
      5. Tap destination tube.
      6. Wait move_settle_delay for pouring animation to complete.

    Args:
        move (Move): The solver move to execute.
        tubes (list[Tube]): List of currently detected Tube objects from vision.
        config (AutomationConfig | None): Timing and dry-run settings.
        emulator (Emulator | None): Active emulator interface instance.

    Returns:
        tuple[bool, str, tuple[int, int], tuple[int, int]]:
            (success, message, source_tap_point, dest_tap_point)
    """
    cfg = config or AutomationConfig()
    em = emulator or Emulator()

    src_tube = find_tube_by_id(tubes, move.from_tube)
    if not src_tube:
        return False, f"Source Tube {move.from_tube} not found in detected geometry.", (0, 0), (0, 0)

    dst_tube = find_tube_by_id(tubes, move.to_tube)
    if not dst_tube:
        return False, f"Destination Tube {move.to_tube} not found in detected geometry.", (0, 0), (0, 0)

    src_tap = get_tube_tap_point(src_tube)
    dst_tap = get_tube_tap_point(dst_tube)

    if cfg.dry_run:
        return True, f"[DRY-RUN] Planned taps: Tube {move.from_tube} {src_tap} -> Tube {move.to_tube} {dst_tap}", src_tap, dst_tap

    # 1. Tap Source Tube
    ok_src, msg_src = tap(src_tap[0], src_tap[1], emulator=em)
    if not ok_src:
        return False, f"Failed to tap source Tube {move.from_tube}: {msg_src}", src_tap, dst_tap

    # 2. Wait between taps
    time.sleep(cfg.tap_delay)

    # 3. Tap Destination Tube
    ok_dst, msg_dst = tap(dst_tap[0], dst_tap[1], emulator=em)
    if not ok_dst:
        return False, f"Failed to tap destination Tube {move.to_tube}: {msg_dst}", src_tap, dst_tap

    return True, f"Tapped Tube {move.from_tube} {src_tap} -> Tube {move.to_tube} {dst_tap}", src_tap, dst_tap


def run_full_execution(
    initial_board: Board,
    initial_tubes: list[Tube],
    moves: list[Move],
    config: AutomationConfig | None = None,
    emulator: Emulator | None = None
) -> ExecutionReport:
    """
    Execute the entire planned move sequence automatically with post-move verification.

    Workflow:
      1. Loops through all planned moves without intermediate user prompts.
      2. For each move:
         - Dispatches source and destination ADB taps.
         - Captures fresh screenshot & runs vision verification.
         - Compares theoretical expected state with perceived state.
         - If verification fails: halts immediately.
         - If actual board is solved: halts immediately with LEVEL CLEARED.
      3. If all moves execute but board is not solved: halts with SOLUTION EXHAUSTED.

    Args:
        initial_board (Board): Starting validated board state.
        initial_tubes (list[Tube]): Starting detected tube geometries.
        moves (list[Move]): Ordered sequence of solver moves.
        config (AutomationConfig | None): Timing and dry-run parameters.
        emulator (Emulator | None): Active emulator interface instance.

    Returns:
        ExecutionReport: Complete outcome summary.
    """
    from solver.search import apply_move
    from solver.visualizer import render_ascii_board
    from automation.verifier import verify_post_move

    cfg = config or AutomationConfig()
    em = emulator or Emulator()
    report = ExecutionReport(total_moves_planned=len(moves))

    if initial_board.is_solved:
        print("\n" + "=" * 55)
        print("  LEVEL CLEARED")
        print("=" * 55)
        print("  Puzzle is already solved.")
        print(f"  Planned Moves  : {len(moves)}")
        print(f"  Executed Moves : 0")
        print("  Status         : ALREADY SOLVED")
        print("  Automation stopped automatically.")
        print("=" * 55 + "\n")
        report.success = True
        return report

    print("\n" + "=" * 55)
    print("  AUTOMATION STARTED")
    print("=" * 55)

    current_board = initial_board.copy()
    current_tubes = list(initial_tubes)

    for step_idx, move in enumerate(moves, start=1):
        expected_board = apply_move(current_board, move)
        src_t = find_tube_by_id(current_tubes, move.from_tube)
        dst_t = find_tube_by_id(current_tubes, move.to_tube)
        src_pt = get_tube_tap_point(src_t) if src_t else (0, 0)
        dst_pt = get_tube_tap_point(dst_t) if dst_t else (0, 0)

        print(f"\nMove {step_idx}/{len(moves)}")
        print(f"Tube {move.from_tube} -> Tube {move.to_tube} | {move.color} x{move.ball_count}")
        print(f"Source tap     : {src_pt}")
        print(f"Destination tap: {dst_pt}")

        ok, msg, _, _ = execute_move(move, current_tubes, config=cfg, emulator=em)
        if not ok:
            print(f"Tap result     : FAILED ({msg})")
            print("\n" + "=" * 55)
            print("  AUTOMATION HALTED")
            print("=" * 55)
            print(f"  Tap dispatch failed at Move {step_idx}.")
            print(f"  Planned Moves  : {len(moves)}")
            print(f"  Executed Moves : {step_idx - 1}")
            print(f"  Status         : TAP DISPATCH ERROR")
            print(f"  Reason         : {msg}")
            print("=" * 55 + "\n")
            report.abort_reason = msg
            return report

        # Dedicated completion verification for the FINAL planned move
        if step_idx == len(moves):
            report.moves_executed = step_idx
            from automation.verifier import verify_final_state
            is_final_ok, final_state, final_msg = verify_final_state(expected_board, emulator=em, config=cfg)
            if is_final_ok:
                print(f"Verification   : PASS ({final_state})")
                print("\n" + "=" * 55)
                print("  LEVEL CLEARED")
                print("=" * 55)
                print("  All planned moves executed successfully.")
                print(f"  Planned Moves  : {len(moves)}")
                print(f"  Executed Moves : {step_idx}")
                print(f"  Final Move     : Tube {move.from_tube} -> Tube {move.to_tube} | {move.color} x{move.ball_count}")
                print(f"  Final State    : {final_state}")
                print(f"  Details        : {final_msg}")
                print("  Verification   : PASS")
                print("  Automation stopped automatically.")
                print("=" * 55 + "\n")
                report.success = True
                return report
            else:
                print(f"Verification   : UNCERTAIN")
                print("\n" + "=" * 55)
                print("  FINAL VERIFICATION UNCERTAIN")
                print("=" * 55)
                print("  Final move was dispatched, but completion could not be confirmed.")
                print(f"  Planned Moves  : {len(moves)}")
                print(f"  Executed Moves : {step_idx}")
                print(f"  Final State    : {final_state}")
                print(f"  Reason         : {final_msg}")
                print("  No further taps will be executed.")
                print("=" * 55 + "\n")
                report.abort_reason = final_msg
                return report

        # Strict verification for intermediate moves (1 .. N-1)
        is_verified, v_msg, actual_b, new_tubes = verify_post_move(expected_board, emulator=em, config=cfg)
        if not is_verified:
            print(f"Verification   : FAIL")
            print("\n" + "=" * 55)
            print("  AUTOMATION HALTED")
            print("=" * 55)
            print(f"  Verification failed after Move {step_idx}.")
            print(f"  Planned Moves  : {len(moves)}")
            print(f"  Executed Moves : {step_idx - 1} (Failed on verification of Move {step_idx})")
            print(f"  Status         : VERIFICATION MISMATCH")
            print(f"\nExpected State:\n{render_ascii_board(expected_board)}\n")
            if actual_b:
                print(f"Actual Perceived State:\n{render_ascii_board(actual_b)}\n")
            print(f"Details: {v_msg}")
            print("No further taps were executed.")
            print("=" * 55 + "\n")
            report.abort_reason = v_msg
            return report

        print(f"Verification   : PASS")
        report.moves_executed = step_idx
        current_board = actual_b
        if new_tubes:
            current_tubes = new_tubes

        if current_board.is_solved:
            print("\n" + "=" * 55)
            print("  LEVEL CLEARED")
            print("=" * 55)
            print("  Puzzle solved successfully.")
            print(f"  Planned Moves  : {len(moves)}")
            print(f"  Executed Moves : {step_idx}")
            print("  Status         : SOLVED")
            print("  Automation stopped automatically.")
            print("=" * 55 + "\n")
            report.success = True
            return report

    if not current_board.is_solved:
        print("\n" + "=" * 55)
        print("  SOLUTION EXHAUSTED")
        print("=" * 55)
        print("  All planned moves were executed, but the board is not solved.")
        print(f"  Planned Moves  : {len(moves)}")
        print(f"  Executed Moves : {report.moves_executed}")
        print("  Status         : UNSOLVED")
        print("  Automation stopped for safety.")
        print(f"\nFinal Detected Board:\n{render_ascii_board(current_board)}\n")
        print("=" * 55 + "\n")
        report.abort_reason = "Solution exhausted without reaching solved state"
        return report

    report.success = True
    return report

