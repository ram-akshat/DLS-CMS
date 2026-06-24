"""
Count Sketch
Authors: Charikar, Chen & Farach-Colton (2002)
Paper: "Finding Frequent Items in Data Streams"

Space: O(k/epsilon^2 * log(1/delta)) where k = heavy-hitter threshold
Uses sign hashing (+1/-1) to reduce bias – estimates can be negative.
"""

import math
import hashlib


class CountSketch:
    def __init__(self, epsilon=0.01, delta=0.01):
        self.epsilon = epsilon
        self.delta = delta
        self.w = math.ceil(3 / (epsilon ** 2))   # width
        self.d = math.ceil(math.log(1 / delta))   # depth
        self.table = [[0] * self.w for _ in range(self.d)]
        self.n = 0

    def _hash(self, item, row):
        """Primary hash → bucket index."""
        h = hashlib.md5(f"h_{item}_{row}".encode()).hexdigest()
        return int(h, 16) % self.w

    def _sign(self, item, row):
        """Sign hash → +1 or -1."""
        h = hashlib.md5(f"s_{item}_{row}".encode()).hexdigest()
        return 1 if int(h, 16) % 2 == 0 else -1

    def update(self, item, count=1):
        self.n += count
        for i in range(self.d):
            col = self._hash(item, i)
            sgn = self._sign(item, i)
            self.table[i][col] += sgn * count

    def query(self, item):
        estimates = []
        for i in range(self.d):
            col = self._hash(item, i)
            sgn = self._sign(item, i)
            estimates.append(sgn * self.table[i][col])
        estimates.sort()
        # Return median
        mid = len(estimates) // 2
        if len(estimates) % 2 == 1:
            return estimates[mid]
        return (estimates[mid - 1] + estimates[mid]) // 2

    def space_bytes(self):
        return self.w * self.d * 4  # 4 bytes per int32 counter

    def space_info(self):
        return {
            "algorithm": "Count Sketch",
            "width (w)": self.w,
            "depth (d)": self.d,
            "total_counters": self.w * self.d,
            "space_bytes": self.space_bytes(),
            "space_kb": round(self.space_bytes() / 1024, 4),
        }


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    random.seed(42)
    words = ["apple", "banana", "cherry", "date", "elderberry",
             "fig", "grape", "honeydew", "kiwi", "lemon"]
    stream = [random.choice(words) for _ in range(10_000)]

    cs = CountSketch(epsilon=0.01, delta=0.01)
    for w in stream:
        cs.update(w)

    print("=== Count Sketch ===")
    info = cs.space_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFrequency estimates (median of signed counts):")
    true_counts = {w: stream.count(w) for w in words}
    for word in words:
        est = cs.query(word)
        true = true_counts[word]
        print(f"  {word:12s}: estimated={est:5d}  true={true:5d}  error={est-true:+d}")
