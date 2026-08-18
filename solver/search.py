"""
File: solver/search.py

Purpose:
    Core move application and Breadth-First Search (BFS) algorithm.
"""

from __future__ import annotations
import time
from collections import deque

from models.board import Board, TubeState
from solver.models import Move, SolverResult
from solver.move_generator import get_valid_moves


def apply_move(board: Board, move: Move) -> Board:
    """
    Apply a Move to an existing Board, returning a new independent Board state.

    Args:
        board (Board): Current board state.
        move (Move): Move to apply.

    Returns:
        Board: New Board state reflecting the move.

    Raises:
        ValueError: If the move is invalid for the given board.
    """
    new_board = board.copy()
    src = new_board.get_tube(move.from_tube)
    dst = new_board.get_tube(move.to_tube)

    if src.is_empty:
        raise ValueError(f"Illegal move: Source Tube {src.id} is empty.")
    if src.top_color != move.color:
        raise ValueError(f"Illegal move: Source Tube {src.id} top color is '{src.top_color}', expected '{move.color}'.")
    if src.ball_count < move.ball_count:
        raise ValueError(f"Illegal move: Source Tube {src.id} only has {src.ball_count} balls, requested {move.ball_count}.")
    if dst.available_space < move.ball_count:
        raise ValueError(f"Illegal move: Dest Tube {dst.id} only has {dst.available_space} slots, requested {move.ball_count}.")
    if not dst.is_empty and dst.top_color != move.color:
        raise ValueError(f"Illegal move: Dest Tube {dst.id} top color '{dst.top_color}' does not match '{move.color}'.")

    # Transfer balls from top of source to top of destination
    for _ in range(move.ball_count):
        ball = src.pop()
        dst.push(ball)

    return new_board


def solve_bfs(
    initial_board: Board,
    max_states: int = 200_000,
    timeout: float = 30.0,
    verbose: bool = False,
    trace: bool = False,
    progress_interval: int = 500
) -> SolverResult:
    """
    Find the shortest sequence of moves to solve the puzzle using Breadth-First Search (BFS).

    Guarantees the optimal (shortest move-count) solution path.

    Args:
        initial_board (Board): The starting board configuration.
        max_states (int): Maximum number of unique board states to explore before giving up.
        timeout (float): Maximum allowed search duration in seconds.
        verbose (bool): If True, prints periodic search exploration progress.
        trace (bool): If True, prints detailed state and legal move expansions.
        progress_interval (int): Frequency of state progress logs in verbose mode.

    Returns:
        SolverResult: Complete search outcome including move sequence and metrics.
    """
    start_time = time.perf_counter()

    if verbose or trace:
        print("\n" + "=" * 55)
        print("  BFS SEARCH DIAGNOSTICS")
        print("=" * 55)
        print(f"  Tubes on Board : {initial_board.num_tubes}")
        print(f"  Total Balls    : {initial_board.total_balls}")
        print(f"  Empty Tubes    : {initial_board.empty_tubes_count}")
        print(f"  Colors ({len(initial_board.colors)})     : {', '.join(sorted(initial_board.colors))}")
        print("-" * 55)

    # 1. Validate initial board
    is_valid, errors = initial_board.validate()
    if not is_valid:
        elapsed = time.perf_counter() - start_time
        if verbose or trace:
            print(f"  [ERROR] Initial board failed validation: {'; '.join(errors)}")
        return SolverResult(
            solved=False,
            states_explored=0,
            elapsed_time=elapsed,
            failure_reason=f"Invalid initial board: {'; '.join(errors)}"
        )

    # 2. Check if already solved
    if initial_board.is_solved:
        elapsed = time.perf_counter() - start_time
        if verbose or trace:
            print("  [INFO] Initial board is already solved (0 moves required).")
        return SolverResult(
            solved=True,
            moves=[],
            move_count=0,
            states_explored=1,
            elapsed_time=elapsed
        )

    # 3. Initialize BFS structures
    initial_key = initial_board.to_state_tuple()
    queue: deque[tuple[Board, tuple[tuple[str, ...], ...]]] = deque([(initial_board, initial_key)])
    visited_states: set[tuple[tuple[str, ...], ...]] = {initial_key}

    # parent_map: child_state_key -> (parent_state_key, move_taken)
    parent_map: dict[tuple[tuple[str, ...], ...], tuple[tuple[tuple[str, ...], ...], Move]] = {}

    states_explored = 0
    solved_key: tuple[tuple[str, ...], ...] | None = None

    if verbose or trace:
        print(f"  [Search Start] Max states: {max_states:,} | Timeout: {timeout:.1f}s")

    # 4. Search Loop
    while queue:
        # Safety checks
        if states_explored >= max_states:
            elapsed = time.perf_counter() - start_time
            if verbose or trace:
                print(f"  [ABORT] State limit exceeded ({states_explored:,} states).")
            return SolverResult(
                solved=False,
                states_explored=states_explored,
                elapsed_time=elapsed,
                failure_reason=f"Exceeded maximum state limit ({max_states:,} states)."
            )

        if time.perf_counter() - start_time >= timeout:
            elapsed = time.perf_counter() - start_time
            if verbose or trace:
                print(f"  [ABORT] Timeout reached ({timeout:.1f}s).")
            return SolverResult(
                solved=False,
                states_explored=states_explored,
                elapsed_time=elapsed,
                failure_reason=f"Search timed out after {timeout:.1f}s ({states_explored:,} states explored)."
            )

        current_board, current_key = queue.popleft()
        states_explored += 1

        if verbose and (states_explored % progress_interval == 0 or states_explored == 1):
            elapsed_now = time.perf_counter() - start_time
            print(f"  [Search] Explored: {states_explored:6,d} | Queue: {len(queue):6,d} | Visited: {len(visited_states):6,d} | Time: {elapsed_now:6.3f}s")

        if current_board.is_solved:
            solved_key = current_key
            break

        # Generate and explore successor states
        valid_moves = get_valid_moves(current_board)

        if trace:
            print(f"\n  [State #{states_explored}] {current_key}")
            print(f"    Legal moves ({len(valid_moves)}): {', '.join(str(m) for m in valid_moves) if valid_moves else 'None'}")

        for move in valid_moves:
            next_board = apply_move(current_board, move)
            next_key = next_board.to_state_tuple()

            if next_key not in visited_states:
                visited_states.add(next_key)
                parent_map[next_key] = (current_key, move)

                if next_board.is_solved:
                    states_explored += 1
                    solved_key = next_key
                    queue.clear()
                    break

                queue.append((next_board, next_key))

    elapsed = time.perf_counter() - start_time

    # 5. Reconstruct solution path
    if solved_key is not None:
        solution_moves: list[Move] = []
        curr = solved_key
        while curr in parent_map:
            prev_state, move_taken = parent_map[curr]
            solution_moves.append(move_taken)
            curr = prev_state

        solution_moves.reverse()

        if verbose or trace:
            print("-" * 55)
            print(f"  [Search Complete] SOLVED in {elapsed:.4f}s")
            print(f"  Explored: {states_explored:,} states | Solution Length: {len(solution_moves)} moves")
            print("=" * 55 + "\n")

        return SolverResult(
            solved=True,
            moves=solution_moves,
            move_count=len(solution_moves),
            states_explored=states_explored,
            elapsed_time=elapsed
        )

    # 6. Unsolvable
    if verbose or trace:
        print("-" * 55)
        print(f"  [Search Complete] UNSOLVED after {states_explored:,} states ({elapsed:.4f}s)")
        print("=" * 55 + "\n")

    return SolverResult(
        solved=False,
        states_explored=states_explored,
        elapsed_time=elapsed,
        failure_reason="No solution exists (all reachable puzzle states exhausted)."
    )
