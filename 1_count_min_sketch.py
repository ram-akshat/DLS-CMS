"""
Count-Min Sketch (CMS)
Authors: Cormode & Muthukrishnan (2005)
Paper: "An Improved Data Stream Summary: The Count-Min Sketch and its Applications"

Space: O(w * d) counters where w = ceil(e/epsilon), d = ceil(ln(1/delta))
"""

import math
import hashlib
import sys


class CountMinSketch:
    def __init__(self, epsilon=0.01, delta=0.01):
        """
        epsilon: error factor (frequency estimate within epsilon * N)
        delta:   failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
        self.w = math.ceil(math.e / epsilon)   # width (columns)
        self.d = math.ceil(math.log(1 / delta)) # depth (rows)
        self.table = [[0] * self.w for _ in range(self.d)]
        self.n = 0  # total items inserted

    def _hash(self, item, row):
        h = hashlib.md5(f"{item}_{row}".encode()).hexdigest()
        return int(h, 16) % self.w

    def update(self, item, count=1):
        self.n += count
        for i in range(self.d):
            col = self._hash(item, i)
            self.table[i][col] += count

    def query(self, item):
        return min(self.table[i][self._hash(item, i)] for i in range(self.d))

    def space_bytes(self):
        """Returns space used in bytes (each counter is a Python int ~ 28 bytes, 
        but for fair comparison we count counter array size as 4 bytes each)."""
        return self.w * self.d * 4  # 4 bytes per int32 counter

    def space_info(self):
        return {
            "algorithm": "Count-Min Sketch",
            "width (w)": self.w,
            "depth (d)": self.d,
            "total_counters": self.w * self.d,
            "space_bytes": self.space_bytes(),
            "space_kb": round(self.space_bytes() / 1024, 4),
        }


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, random, string

    random.seed(42)
    words = ["apple", "banana", "cherry", "date", "elderberry",
             "fig", "grape", "honeydew", "kiwi", "lemon"]
    stream = [random.choice(words) for _ in range(10_000)]

    cms = CountMinSketch(epsilon=0.01, delta=0.01)
    for w in stream:
        cms.update(w)

    print("=== Count-Min Sketch ===")
    info = cms.space_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFrequency estimates:")
    true_counts = {w: stream.count(w) for w in words}
    for word in words:
        est = cms.query(word)
        true = true_counts[word]
        print(f"  {word:12s}: estimated={est:5d}  true={true:5d}  error={est-true:+d}")
