from __future__ import annotations
from models.board import Board
from solver.models import Move, SolverResult
from solver.search import solve_bfs, apply_move
from solver.visualizer import render_ascii_board, render_step_transition


def solve(
    board: Board,
    max_states: int = 200_000,
    timeout: float = 30.0,
    verbose: bool = False,
    trace: bool = False,
    progress_interval: int = 500
) -> SolverResult:
    """
    Solve a Ball Sort puzzle board.

    Args:
        board (Board): The formal puzzle board to solve.
        max_states (int): State exploration threshold.
        timeout (float): Time limit in seconds.
        verbose (bool): If True, prints periodic search exploration progress.
        trace (bool): If True, prints detailed state and move expansions.
        progress_interval (int): Frequency of state progress logs in verbose mode.

    Returns:
        SolverResult: Complete solving results including optimal moves,
                      states explored, and elapsed time.
    """
    return solve_bfs(
        board,
        max_states=max_states,
        timeout=timeout,
        verbose=verbose,
        trace=trace,
        progress_interval=progress_interval
    )


def validate_and_replay_solution(
    initial_board: Board,
    moves: list[Move]
) -> tuple[bool, str, Board]:
    """
    Replay a solution move sequence step-by-step from the initial board state.

    Verifies that:
      1. Every move in the sequence is currently legal.
      2. Board state transitions are valid.
      3. Total ball count is strictly conserved.
      4. Per-color counts are strictly conserved.
      5. The final board state reaches board.is_solved == True.

    Args:
        initial_board (Board): Starting board configuration.
        moves (list[Move]): Sequence of moves to execute.

    Returns:
        tuple[bool, str, Board]: (success, status_message, final_board_state)
    """
    current_board = initial_board.copy()
    init_total_balls = initial_board.total_balls
    init_color_counts = initial_board.color_counts

    for step_num, move in enumerate(moves, start=1):
        try:
            current_board = apply_move(current_board, move)
        except Exception as e:
            return False, f"Replay failed at move {step_num} ({move}): {e}", current_board

        # Invariant checks
        if current_board.total_balls != init_total_balls:
            return False, f"Ball count mismatch after move {step_num}: {current_board.total_balls} != {init_total_balls}", current_board

        if current_board.color_counts != init_color_counts:
            return False, f"Color count mismatch after move {step_num}: {current_board.color_counts} != {init_color_counts}", current_board

    if not current_board.is_solved:
        return False, "Replay completed all moves but final board is NOT solved.", current_board

    return True, f"Replay verified successfully ({len(moves)} moves reached solved state).", current_board


def replay_and_visualize_solution(
    initial_board: Board,
    moves: list[Move],
    print_steps: bool = True
) -> tuple[bool, str, list[Board]]:
    """
    Replay a solution in memory and print visual step-by-step ASCII board states.

    Args:
        initial_board (Board): Starting board configuration.
        moves (list[Move]): Solution moves to simulate.
        print_steps (bool): If True, prints ASCII diagrams after each move.

    Returns:
        tuple[bool, str, list[Board]]: (success, message, history_of_boards)
    """
    history: list[Board] = [initial_board.copy()]
    current_board = initial_board.copy()
    total_moves = len(moves)

    if print_steps:
        print("\n" + "=" * 60)
        print("  IN-MEMORY SOLUTION REPLAY & VISUALIZATION")
        print("=" * 60)
        print("  INITIAL BOARD STATE:")
        print(render_ascii_board(current_board))
        print("=" * 60)

    init_total_balls = initial_board.total_balls
    init_color_counts = initial_board.color_counts

    for step_num, move in enumerate(moves, start=1):
        try:
            current_board = apply_move(current_board, move)
            history.append(current_board.copy())
        except Exception as e:
            err = f"Replay failed at move {step_num} ({move}): {e}"
            if print_steps:
                print(f"\n[ERROR] {err}\n")
            return False, err, history

        # Invariant checks
        if current_board.total_balls != init_total_balls:
            err = f"Ball count mismatch after move {step_num}"
            return False, err, history
        if current_board.color_counts != init_color_counts:
            err = f"Color count mismatch after move {step_num}"
            return False, err, history

        if print_steps:
            print(render_step_transition(step_num, total_moves, move, current_board))

    if not current_board.is_solved:
        err = "Replay finished but final board is NOT solved."
        return False, err, history

    if print_steps:
        print("=" * 60)
        print("  FINAL BOARD STATE — PUZZLE SOLVED!")
        print("=" * 60)
        print(render_ascii_board(current_board))
        print(f"\n  Summary: All {total_moves} moves verified in memory.")
        print(f"  Ball count conserved ({init_total_balls} balls).")
        print(f"  Color distribution conserved ({len(init_color_counts)} colors).")
        print("=" * 60 + "\n")

    return True, f"Replay succeeded ({total_moves} moves).", history
