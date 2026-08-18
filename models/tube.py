from dataclasses import dataclass, field
import numpy as np


@dataclass
class Tube:
    id: int
    x: int
    y: int
    width: int
    height: int
    area: float
    contour: np.ndarray
    balls: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    balls_present: list = field(default_factory=list)

    @property
    def center(self):
        return (
            self.x + self.width // 2,
            self.y + self.height // 2
        )

    @property
    def aspect_ratio(self):
        return self.height / self.width

    @property
    def capacity(self):
        if self.slots:
            return len(self.slots)
        if self.balls:
            return len(self.balls)
        return 0

    @property
    def ball_count(self):
        if self.balls_present:
            return sum(1 for b in self.balls_present if b)
        if self.balls:
            return sum(1 for b in self.balls if b and b != "EMPTY")
        return 0

    @property
    def is_empty(self):
        return self.ball_count == 0

    @property
    def is_full(self):
        return self.capacity > 0 and self.ball_count == self.capacity