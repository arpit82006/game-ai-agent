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

    @property
    def center(self):
        return (
            self.x + self.width // 2,
            self.y + self.height // 2
        )

    @property
    def aspect_ratio(self):
        return self.height / self.width