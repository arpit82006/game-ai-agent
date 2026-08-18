"""
File: models/board.py

Purpose:
    Formal solver-ready representation of the Ball Sort puzzle state.
    Completely decoupled from OpenCV, image coordinates, ADB, and emulator details.

Conventions:
    - Tube ordering: Deterministic 1-based indexing (Tube 1 .. Tube N).
    - Ball ordering: Top-to-Bottom.
        * tube.balls[0]  = TOP ball (exposed, movable).
        * tube.balls[-1] = BOTTOM ball (resting on the bottom of the tube).
        * Empty tube     = [] (length 0).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import copy


@dataclass
class TubeState:
    """
    Independent representation of a single tube's state in the puzzle.

    Attributes:
        id (int): Deterministic tube identifier (1-indexed).
        capacity (int): Maximum ball capacity of this specific tube.
        balls (list[str]): List of color names ordered from TOP to BOTTOM.
    """
    id: int
    capacity: int
    balls: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Ensure balls is always a mutable list of strings
        if isinstance(self.balls, tuple):
            self.balls = list(self.balls)

    @property
    def ball_count(self) -> int:
        """Number of balls currently inside the tube."""
        return len(self.balls)

    @property
    def available_space(self) -> int:
        """Number of empty slots available in this tube."""
        return max(0, self.capacity - len(self.balls))

    @property
    def is_empty(self) -> bool:
        """True if the tube contains zero balls."""
        return len(self.balls) == 0

    @property
    def is_full(self) -> bool:
        """True if the tube contains balls equal to its capacity."""
        return len(self.balls) >= self.capacity

    @property
    def is_pure(self) -> bool:
        """
        True if all balls in the tube have the exact same known color (or if empty).
        Tubes containing GRAY mystery balls are never pure.
        """
        if any(b == "GRAY" for b in self.balls):
            return False
        if len(self.balls) <= 1:
            return True
        first = self.balls[0]
        return all(b == first for b in self.balls)

    @property
    def is_solved(self) -> bool:
        """
        A tube is considered solved if:
          1. It is completely empty, OR
          2. It is completely full AND contains only one single uniform color.
        Tubes with unrevealed GRAY mystery balls are never solved.
        """
        if self.is_empty:
            return True
        return self.is_full and self.is_pure

    @property
    def top_color(self) -> str | None:
        """Color of the topmost ball (or None if tube is empty)."""
        return self.balls[0] if self.balls else None

    @property
    def top_same_color_count(self) -> int:
        """
        Number of contiguous balls matching the topmost color from top down.
        E.g., ['RED', 'RED', 'BLUE'] -> 2
              ['RED'] -> 1
              [] -> 0
        """
        if not self.balls:
            return 0
        target = self.balls[0]
        count = 0
        for b in self.balls:
            if b == target:
                count += 1
            else:
                break
        return count

    def push(self, color: str) -> None:
        """
        Place a ball on top of the tube stack.
        """
        if self.is_full:
            raise ValueError(f"Cannot push to full Tube {self.id} (capacity {self.capacity})")
        self.balls.insert(0, color)

    def pop(self) -> str:
        """
        Remove and return the topmost ball.
        """
        if self.is_empty:
            raise ValueError(f"Cannot pop from empty Tube {self.id}")
        return self.balls.pop(0)

    def copy(self) -> TubeState:
        """Return an independent deep copy of this tube state."""
        return TubeState(id=self.id, capacity=self.capacity, balls=list(self.balls))

    def to_list(self) -> list[str]:
        """Return a copy of the ball list (TOP to BOTTOM)."""
        return list(self.balls)

    def to_tuple(self) -> tuple[str, ...]:
        """Return an immutable tuple of the ball list."""
        return tuple(self.balls)

    def __repr__(self) -> str:
        balls_str = " -> ".join(self.balls) if self.balls else "EMPTY"
        return f"TubeState(id={self.id}, cap={self.capacity}, count={self.ball_count}, balls=[{balls_str}])"


@dataclass
class Board:
    """
    Formal, solver-ready representation of the full Ball Sort board.

    Attributes:
        tubes (list[TubeState]): Ordered list of TubeState objects.
    """
    tubes: list[TubeState] = field(default_factory=list)

    @property
    def num_tubes(self) -> int:
        """Total number of tubes on the board."""
        return len(self.tubes)

    @property
    def total_balls(self) -> int:
        """Total number of balls across all tubes."""
        return sum(t.ball_count for t in self.tubes)

    @property
    def empty_tubes_count(self) -> int:
        """Number of empty tubes."""
        return sum(1 for t in self.tubes if t.is_empty)

    @property
    def colors(self) -> set[str]:
        """Set of all distinct color names present on the board."""
        all_colors = set()
        for t in self.tubes:
            all_colors.update(t.balls)
        return all_colors

    @property
    def color_counts(self) -> dict[str, int]:
        """Dictionary mapping each color name to its total ball count."""
        counts = {}
        for t in self.tubes:
            for b in t.balls:
                counts[b] = counts.get(b, 0) + 1
        return counts

    @property
    def has_mystery_balls(self) -> bool:
        """True if the board contains unrevealed GRAY mystery balls."""
        return "GRAY" in self.colors

    @property
    def mystery_ball_count(self) -> int:
        """Number of unrevealed GRAY mystery balls on the board."""
        return self.color_counts.get("GRAY", 0)

    @property
    def known_ball_count(self) -> int:
        """Number of revealed known-color balls on the board."""
        return self.total_balls - self.mystery_ball_count

    @property
    def is_solved(self) -> bool:
        """
        True if all tubes on the board satisfy the solved condition
        (every tube is either empty or full and monochromatic).
        Boards containing unrevealed GRAY mystery balls are never solved.
        """
        if not self.tubes or self.total_balls == 0 or self.has_mystery_balls:
            return False
        return all(t.is_solved for t in self.tubes)

    def get_tube(self, tube_id: int) -> TubeState:
        """Get a tube by its 1-based ID."""
        for t in self.tubes:
            if t.id == tube_id:
                return t
        raise KeyError(f"Tube {tube_id} not found on board.")

    def __getitem__(self, index: int) -> TubeState:
        """Zero-based indexing into the tubes collection."""
        return self.tubes[index]

    def __iter__(self):
        """Iterate over all tubes on the board."""
        return iter(self.tubes)

    def __len__(self) -> int:
        """Number of tubes on the board."""
        return len(self.tubes)

    def copy(self) -> Board:
        """Return an independent deep copy of the board."""
        return Board(tubes=[t.copy() for t in self.tubes])

    def to_lists(self) -> list[list[str]]:
        """
        Convert board state into a plain Python list of lists of colors.
        Each inner list is ordered from TOP to BOTTOM.
        """
        return [t.to_list() for t in self.tubes]

    def to_state_tuple(self) -> tuple[tuple[str, ...], ...]:
        """
        Return an immutable, hashable tuple of tuples representing the board.
        Essential for solver visited-state sets and memoization.
        """
        return tuple(t.to_tuple() for t in self.tubes)

    def to_dict(self) -> dict:
        """Serialize board structure to a standard dictionary."""
        return {
            "num_tubes": self.num_tubes,
            "total_balls": self.total_balls,
            "empty_tubes": self.empty_tubes_count,
            "color_counts": self.color_counts,
            "has_mystery_balls": self.has_mystery_balls,
            "mystery_balls": self.mystery_ball_count,
            "known_balls": self.known_ball_count,
            "tubes": [
                {
                    "id": t.id,
                    "capacity": t.capacity,
                    "ball_count": t.ball_count,
                    "balls": t.to_list()
                }
                for t in self.tubes
            ]
        }

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate logical consistency and puzzle invariants.

        Checks:
          1. At least 1 tube exists.
          2. Every tube has positive capacity (> 0).
          3. No tube contains more balls than its capacity.
          4. No invalid/empty/None/BLACK color strings in active balls.
          5. Positive total ball count.
          6. All color counts > 0.
          7. Total balls do not exceed total board capacity.

        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_error_messages)
        """
        errors = []

        if self.num_tubes == 0:
            errors.append("Board contains zero tubes.")
            return False, errors

        total_cap = 0
        for t in self.tubes:
            if t.capacity <= 0:
                errors.append(f"Tube {t.id} has invalid capacity: {t.capacity} (must be > 0).")
            total_cap += max(0, t.capacity)

            if len(t.balls) > t.capacity:
                errors.append(f"Tube {t.id} contains {len(t.balls)} balls, exceeding capacity {t.capacity}.")

            for idx, ball in enumerate(t.balls):
                if not ball or ball.strip() == "" or ball in ("EMPTY", "BLACK"):
                    errors.append(f"Tube {t.id} ball at position {idx} has invalid color marker: {repr(ball)}. BLACK/EMPTY must represent empty space, not an active ball.")

        if self.total_balls == 0:
            errors.append("Board contains zero balls (entire board is empty).")

        if self.total_balls > total_cap:
            errors.append(f"Total balls ({self.total_balls}) exceeds total board capacity ({total_cap}).")

        is_valid = (len(errors) == 0)
        return is_valid, errors

    @classmethod
    def from_lists(cls, lists: Sequence[Sequence[str]], capacities: Sequence[int] | int = 4) -> Board:
        """
        Construct a Board from plain nested lists of color names.

        Empty space markers 'BLACK' and 'EMPTY' are treated as empty slots.
        Mystery balls 'GRAY' are treated as occupied balls with hidden color.

        Args:
            lists: Nested sequence where each sub-sequence lists balls from TOP to BOTTOM.
            capacities: Single int capacity for all tubes, or list of per-tube capacities.

        Returns:
            Board: Populated board instance.
        """
        tubes = []
        for i, b_list in enumerate(lists, start=1):
            if isinstance(capacities, (int, float)):
                cap = int(capacities)
            else:
                cap = capacities[i - 1] if i - 1 < len(capacities) else 4
            active_balls = [str(c) for c in b_list if c and c not in ("EMPTY", "BLACK")]
            tubes.append(TubeState(id=i, capacity=cap, balls=active_balls))
        return cls(tubes=tubes)

    @classmethod
    def from_vision_tubes(cls, vision_tubes: Sequence[object]) -> Board:
        """
        Convert a sequence of OpenCV Tube objects from the vision pipeline into a formal Board.

        Preserves individual tube capacities, tube reading IDs, and top-to-bottom ball ordering.

        Args:
            vision_tubes: List of Tube objects from vision.detect_tubes / pipeline.

        Returns:
            Board: Populated Board model.
        """
        board_tubes = []
        for vt in vision_tubes:
            # vt.capacity is dynamically computed from that tube's own geometry
            cap = getattr(vt, "capacity", 4)
            if cap <= 0:
                cap = 4

            # vt.balls contains color strings (or "EMPTY" for unoccupied slots) from top to bottom
            raw_balls = getattr(vt, "balls", [])
            active_balls = [str(c) for c in raw_balls if c and c != "EMPTY"]

            t_id = getattr(vt, "id", len(board_tubes) + 1)
            board_tubes.append(TubeState(id=t_id, capacity=cap, balls=active_balls))

        return cls(tubes=board_tubes)

    def formatted_summary(self) -> str:
        """Return a clean human-readable board summary."""
        lines = []
        for t in self.tubes:
            if t.is_empty:
                lines.append(f"  Tube {t.id:2d} (capacity {t.capacity}): [EMPTY TUBE]")
            else:
                balls_desc = " -> ".join(t.balls)
                empty_s = t.available_space
                lines.append(f"  Tube {t.id:2d} (capacity {t.capacity}): [TOP] {balls_desc} [BOTTOM]  ({t.ball_count}/{t.capacity} balls, {empty_s} empty)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board(num_tubes={self.num_tubes}, total_balls={self.total_balls}, tubes={self.to_lists()})"
