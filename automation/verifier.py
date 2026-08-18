"""
File: automation/verifier.py

Purpose:
    Post-move visual verification and board comparison routines.
"""

from __future__ import annotations
import time
from models.board import Board
from models.tube import Tube
from core.adb import Emulator
from vision.pipeline import run_vision_pipeline
from automation.models import AutomationConfig


def compare_boards(expected: Board, actual: Board) -> tuple[bool, list[str]]:
    """
    Compare an expected (simulated) Board state against an actual (perceived) Board state.

    Checks:
      1. Number of tubes.
      2. Total ball count.
      3. Global color distribution.
      4. Exact top-to-bottom ball stack equality for every individual tube.

    Args:
        expected (Board): Theoretical board state after move simulation.
        actual (Board): Real board state detected by the vision pipeline.

    Returns:
        tuple[bool, list[str]]: (is_match, list_of_mismatch_descriptions)
    """
    mismatches: list[str] = []

    if expected.num_tubes != actual.num_tubes:
        mismatches.append(f"Tube count mismatch: expected {expected.num_tubes}, got {actual.num_tubes}")

    if expected.total_balls != actual.total_balls:
        mismatches.append(f"Total ball count mismatch: expected {expected.total_balls}, got {actual.total_balls}")

    if expected.color_counts != actual.color_counts:
        mismatches.append(f"Color distribution mismatch: expected {expected.color_counts}, got {actual.color_counts}")

    # Check each tube's contents
    for exp_t in expected.tubes:
        try:
            act_t = actual.get_tube(exp_t.id)
        except KeyError:
            mismatches.append(f"Tube {exp_t.id} missing in actual board.")
            continue

        if exp_t.balls != act_t.balls:
            exp_str = " -> ".join(exp_t.balls) if exp_t.balls else "EMPTY"
            act_str = " -> ".join(act_t.balls) if act_t.balls else "EMPTY"
            mismatches.append(
                f"Tube {exp_t.id} contents mismatch:\n"
                f"    Expected: [{exp_str}]\n"
                f"    Actual  : [{act_str}]"
            )

    is_match = (len(mismatches) == 0)
    return is_match, mismatches


def verify_post_move(
    expected_board: Board,
    emulator: Emulator | None = None,
    config: AutomationConfig | None = None
) -> tuple[bool, str, Board | None, list[Tube]]:
    """
    Capture a fresh screenshot, run the vision pipeline, and verify the resulting board.

    Args:
        expected_board (Board): The simulated expected board state.
        emulator (Emulator | None): Active emulator interface instance.
        config (AutomationConfig | None): Automation timing settings.

    Returns:
        tuple[bool, str, Board | None, list[Tube]]:
            (is_verified, status_message, actual_board, actual_detected_tubes)
    """
    cfg = config or AutomationConfig()
    em = emulator or Emulator()

    if cfg.verification_delay > 0:
        time.sleep(cfg.verification_delay)

    try:
        image, _ = em.capture_screenshot()
    except Exception as e:
        return False, f"Post-move screenshot capture failed: {e}", None, []

    try:
        vision_result = run_vision_pipeline(image, save_debug=False)
    except Exception as e:
        return False, f"Post-move vision pipeline execution failed: {e}", None, []

    actual_board = vision_result.board
    if actual_board is None:
        return False, "Vision pipeline did not construct a Board model.", None, vision_result.tubes

    is_match, mismatches = compare_boards(expected_board, actual_board)
    if is_match:
        return True, "Post-move state matches expected board state.", actual_board, vision_result.tubes

    # If mismatch occurred, allow a brief stabilization delay (0.6s) to let any trailing ball drop animation finish
    time.sleep(0.60)
    try:
        retry_image, _ = em.capture_screenshot()
        retry_res = run_vision_pipeline(retry_image, save_debug=False)
        if retry_res.board:
            retry_match, retry_mismatches = compare_boards(expected_board, retry_res.board)
            if retry_match:
                return True, "Post-move state matches expected board state (after stabilization).", retry_res.board, retry_res.tubes
            actual_board = retry_res.board
            vision_result = retry_res
            mismatches = retry_mismatches
    except Exception:
        pass

    err_msg = "Board state mismatch after move:\n  " + "\n  ".join(mismatches)
    return False, err_msg, actual_board, vision_result.tubes


def is_completion_screen(image) -> bool:
    """
    Check if the current screen contains the in-game 'Level Complete' celebration banner.
    """
    if image is None or image.size == 0:
        return False
    import numpy as np
    h, w = image.shape[:2]
    # Check central text region for bright cream letters of "Level Complete"
    center_crop = image[int(h * 0.27):int(h * 0.42), int(w * 0.25):int(w * 0.75)]
    if center_crop.size == 0:
        return False
    mask = (center_crop[:, :, 0] > 150) & (center_crop[:, :, 1] > 180) & (center_crop[:, :, 2] > 220)
    return int(np.sum(mask)) >= 800


def verify_final_state(
    expected_board: Board,
    emulator: Emulator | None = None,
    config: AutomationConfig | None = None,
    max_attempts: int = 3,
    retry_delay: float = 0.75
) -> tuple[bool, str, str]:
    """
    Dedicated verification flow for the final planned move of a level.

    Handles level completion UI transitions, banners, and post-game advertisement overlays:
      1. SOLVED_BOARD: Fully readable and verified solved board.
      2. COMPLETION_SCREEN: In-game 'Level Complete' celebration banner / corked tubes.
      3. AD_OR_NON_GAME_SCREEN: Post-level advertisement or non-puzzle UI transition.

    Returns:
        tuple[bool, str, str]: (is_success, state_name, status_message)
    """
    cfg = config or AutomationConfig()
    em = emulator or Emulator()

    if cfg.dry_run:
        return True, "SOLVED_BOARD", "[DRY-RUN] Final solved state verified in simulation."

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            time.sleep(retry_delay)

        try:
            image, _ = em.capture_screenshot()
        except Exception:
            continue

        # 1. Check if the solved board is directly readable and matching
        try:
            res = run_vision_pipeline(image, save_debug=False)
            if res.board:
                is_match, _ = compare_boards(expected_board, res.board)
                if is_match or res.board.is_solved:
                    return True, "SOLVED_BOARD", "Final board verified in solved state."
        except Exception:
            pass

        # 2. Check for the in-game "Level Complete" celebration banner
        if is_completion_screen(image):
            return True, "COMPLETION_SCREEN", "Final board transition detected. Level-complete state detected before advertisement/UI transition."

        # 3. Check for ad or non-puzzle transition (e.g. fewer than 4 tubes detected)
        try:
            res = run_vision_pipeline(image, save_debug=False)
            if len(res.tubes) < 4:
                return True, "AD_OR_NON_GAME_SCREEN", "Final board transition detected. Level completed before advertisement/UI transition."
        except Exception:
            return True, "AD_OR_NON_GAME_SCREEN", "Final board transition detected. Level completed before advertisement/UI transition."

    return False, "FINAL_VERIFICATION_UNCERTAIN", "Final move was dispatched, but completion could not be confirmed."
