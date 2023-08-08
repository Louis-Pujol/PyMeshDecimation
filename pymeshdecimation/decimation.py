"""Class decimation"""


import numpy as np
from typing import Optional


class Decimation:
    def __init__(
        self, target_reduction: Optional[float] = None, n_points: Optional[int] = None
    ) -> None:
        if target_reduction is None and n_points is None:
            raise ValueError("Either target_reduction or n_points must be specified")
        if target_reduction is not None and n_points is not None:
            raise ValueError(
                "Only one of target_reduction or n_points must be specified"
            )

        if target_reduction is not None:
            assert 0 < target_reduction < 1, "target_reduction must be in (0, 1)"
        if n_points is not None:
            assert n_points > 0, "n_points must be positive"

        self.target_reduction = target_reduction
        self.n_points = n_points
