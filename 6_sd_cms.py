"""
SD-CMS: Signed Decay Count-Min Sketch
======================================
Supports : Insert, Delete, Query, Space reporting
Model    : Exponential time decay (sliding window approximation)
Estimator: Median of signed counters (unbiased)

Comparison-ready version
"""

import hashlib
import math
import statistics


class SDCMS:
    def __init__(self, epsilon: float = 0.01, delta: float = 0.01,
                 lambda_: float = 0.9):

        if not (0 < lambda_ < 1):
            raise ValueError("lambda_ must be strictly between 0 and 1")
        if epsilon <= 0 or delta <= 0:
            raise ValueError("epsilon and delta must be positive")

        self.epsilon = epsilon
        self.delta   = delta
        self.lambda_ = lambda_

        # Total decayed weight (geometric series)
        self.F = 1.0 / (1.0 - lambda_)

        # Dimensions
        self.w = math.ceil(math.e * self.F / epsilon)
        self.d = math.ceil(math.log(1.0 / delta))

        # Counter matrix
        self.C = [[0.0] * self.w for _ in range(self.d)]

        # Logical clock
        self._time = 0
        self._last_decay_time = 0

        # Hash seeds
        self._seeds_pos  = [i * 2654435761 for i in range(1, self.d + 1)]
        self._seeds_sign = [i * 2246822519 for i in range(1, self.d + 1)]

    # ─────────────────────────────────────────────
    # Hash Functions
    # ─────────────────────────────────────────────

    def _hash_position(self, item: str, row: int) -> int:
        seed = self._seeds_pos[row]
        raw = int(hashlib.md5(f"{seed}:{item}".encode()).hexdigest(), 16)
        return raw % self.w

    def _hash_sign(self, item: str, row: int) -> int:
        seed = self._seeds_sign[row]
        raw = int(hashlib.md5(f"{seed}:{item}".encode()).hexdigest(), 16)
        return 1 if (raw % 2 == 0) else -1

    # ─────────────────────────────────────────────
    # Decay Logic (CORE of your contribution)
    # ─────────────────────────────────────────────

    def _apply_decay(self, new_time: int):
        elapsed = new_time - self._last_decay_time
        if elapsed <= 0:
            return

        alpha = self.lambda_ ** elapsed

        for i in range(self.d):
            for j in range(self.w):
                self.C[i][j] *= alpha

        self._last_decay_time = new_time

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def update(self, item: str, count: float = 1.0):
        """
        Insert (count > 0) or delete (count < 0)
        """
        self._time += 1

        # Step 1: Apply decay
        self._apply_decay(self._time)

        # Step 2: Signed update
        for i in range(self.d):
            col  = self._hash_position(item, i)
            sign = self._hash_sign(item, i)
            self.C[i][col] += sign * count

    def query(self, item: str) -> float:
        """
        Median of signed estimates → unbiased
        """
        estimates = []
        for i in range(self.d):
            col  = self._hash_position(item, i)
            sign = self._hash_sign(item, i)
            estimates.append(sign * self.C[i][col])

        return statistics.median(estimates)

    # ─────────────────────────────────────────────
    # Space Reporting (CRITICAL for your paper)
    # ─────────────────────────────────────────────

    def space_bytes(self) -> int:
        return self.d * self.w * 8  # float64

    def space_info(self) -> dict:
        return {
            "algorithm": "SD_CMS",
            "lambda": self.lambda_,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "F": round(self.F, 4),
            "width (w)": self.w,
            "depth (d)": self.d,
            "total_counters": self.w * self.d,
            "space_bytes": self.space_bytes(),
            "space_kb": round(self.space_bytes() / 1024, 4),
        }

    def get_sketch_info(self):
        return self.space_info()