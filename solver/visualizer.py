"""
File: solver/visualizer.py

Purpose:
    Clean ASCII visualization and formatting utilities for Ball Sort puzzle boards.
    Works in any standard terminal without requiring external dependencies or emojis.
"""

from __future__ import annotations
from models.board import Board, TubeState
from solver.models import Move

# Standard color abbreviations for clean fixed-width column display
COLOR_ABBR = {
    "RED": "RED",
    "ORANGE": "ORG",
    "YELLOW": "YEL",
    "GREEN": "GRN",
    "EMERALD_GREEN": "EMG",
    "LIGHT_BLUE": "LBL",
    "DARK_BLUE": "DBL",
    "DARK_PURPLE": "DPU",
    "MAGENTA": "MAG",
    "PINK": "PNK",
    "GRAY": " ? ",
    "BLACK": " . ",
}


def get_color_abbr(color: str | None) -> str:
    """Return a fixed 3-character abbreviation for a color name."""
    if not color:
        return " . "
    if color in COLOR_ABBR:
        return COLOR_ABBR[color]
    # Generic fallback for future / special colors
    return (color[:3] if len(color) >= 3 else color.ljust(3)).upper()


def render_ascii_board(board: Board, title: str | None = None) -> str:
    """
    Render a vertical ASCII column diagram of all tubes on the board.

    Example layout:
        T1     T2     T3     T4     T5     T6     T7
       ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
       │EMG│  │MAG│  │EMG│  │EMG│  │YEL│  │ . │  │ . │
       │EMG│  │LBL│  │YEL│  │YEL│  │YEL│  │ . │  │ . │
       │DPU│  │LBL│  │DPU│  │DPU│  │LBL│  │ . │  │ . │
       │MAG│  │DPU│  │MAG│  │MAG│  │LBL│  │ . │  │ . │
       └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘
       (4/4)  (4/4)  (4/4)  (4/4)  (4/4)  (0/4)  (0/4)

    Args:
        board (Board): The board to render.
        title (str | None): Optional section title.

    Returns:
        str: Multi-line string representation.
    """
    if not board.tubes:
        return "[Empty Board — No Tubes]"

    max_capacity = max(t.capacity for t in board.tubes)
    num_tubes = len(board.tubes)

    lines: list[str] = []
    if title:
        lines.append(f"--- {title} ---")

    # Header with Tube IDs
    headers = [f" T{t.id:<2d} " for t in board.tubes]
    lines.append("  " + "  ".join(headers))

    # Top lip of tubes
    lips = ["┌───┐" for _ in range(num_tubes)]
    lines.append("  " + "  ".join(lips))

    # Render each vertical level from TOP (slot 0) down to BOTTOM (slot max_capacity - 1)
    for slot_idx in range(max_capacity):
        row_cells = []
        for tube in board.tubes:
            empty_slots = tube.capacity - tube.ball_count
            if slot_idx < empty_slots:
                # Empty slot
                row_cells.append("│ . │")
            else:
                # Occupied slot (balls[0] is topmost occupied ball)
                ball_idx = slot_idx - empty_slots
                if ball_idx < len(tube.balls):
                    color_name = tube.balls[ball_idx]
                    abbr = get_color_abbr(color_name)
                    row_cells.append(f"│{abbr}│")
                else:
                    row_cells.append("│ . │")
        lines.append("  " + "  ".join(row_cells))

    # Bottom base of tubes
    bases = ["└───┘" for _ in range(num_tubes)]
    lines.append("  " + "  ".join(bases))

    # Capacity and fill indicators
    counts = [f"({t.ball_count}/{t.capacity})" for t in board.tubes]
    lines.append("  " + "  ".join(f"{c:^5s}" for c in counts))

    return "\n".join(lines)


def render_step_transition(
    step_num: int,
    total_steps: int,
    move: Move,
    board_after: Board
) -> str:
    """
    Format a diagnostic move transition showing the action taken and resulting board.

    Args:
        step_num (int): Current step index (1-based).
        total_steps (int): Total number of moves in sequence.
        move (Move): Move just executed.
        board_after (Board): Board state after applying the move.

    Returns:
        str: Formatted multi-line diagnostic block.
    """
    lines = []
    separator = "=" * 60
    lines.append("\n" + separator)
    lines.append(f"  MOVE {step_num:2d} / {total_steps:2d} : {move}")
    lines.append(separator)
    lines.append(render_ascii_board(board_after))
    lines.append("-" * 60)
    lines.append("  Tubes state (TOP -> BOTTOM):")
    for t in board_after.tubes:
        if t.is_empty:
            lines.append(f"    Tube {t.id:2d} (cap {t.capacity}): [EMPTY]")
        else:
            balls_str = " -> ".join(t.balls)
            lines.append(f"    Tube {t.id:2d} (cap {t.capacity}): [TOP] {balls_str} [BOTTOM] ({t.ball_count}/{t.capacity})")
    lines.append(separator + "\n")
    return "\n".join(lines)
