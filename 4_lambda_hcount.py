"""
λ-HCount (Lambda-HCount)
Authors: Chen & Mei (2009)
Paper: "HCount: A Fast Frequency Estimation Algorithm for Skewed Data Streams"

λ-HCount combines a heavy-hitter exact table for frequent items
with a Count-Min-Sketch residual for the rest.
The λ parameter controls the size of the exact (heavy) table.

Space: O(lambda) for exact table  +  O(w * d) for residual CMS
"""

import math
import hashlib


class LambdaHCount:
    def __init__(self, epsilon=0.01, delta=0.01, lambda_=64):
        """
        epsilon, delta : CMS accuracy parameters for residual sketch
        lambda_        : max number of heavy-hitter slots (exact table size)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.lambda_ = lambda_

        # Exact heavy-hitter table  {item -> count}
        self.heavy = {}

        # Residual CMS for non-heavy items
        self.w = math.ceil(math.e / epsilon)
        self.d = math.ceil(math.log(1 / delta))
        self.cms = [[0] * self.w for _ in range(self.d)]

        self.n = 0  # stream length

    # ── Hash helpers ─────────────────────────────────────────────────────────

    def _cms_hash(self, item, row):
        h = hashlib.md5(f"{item}_{row}".encode()).hexdigest()
        return int(h, 16) % self.w

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, item, count=1):
        self.n += count

        if item in self.heavy:
            self.heavy[item] += count
        elif len(self.heavy) < self.lambda_:
            # Slot available – track exactly
            self.heavy[item] = count
        else:
            # No slot – fall through to residual CMS
            for i in range(self.d):
                self.cms[i][self._cms_hash(item, i)] += count

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, item):
        if item in self.heavy:
            return self.heavy[item]
        return min(self.cms[i][self._cms_hash(item, i)] for i in range(self.d))

    # ── Space ─────────────────────────────────────────────────────────────────

    def space_bytes(self):
        # Heavy table: each slot stores (item_key + count)
        # Assume 32-byte key + 4-byte int per entry
        heavy_bytes = len(self.heavy) * (32 + 4)
        cms_bytes = self.w * self.d * 4
        return heavy_bytes + cms_bytes

    def space_info(self):
        return {
            "algorithm": "λ-HCount",
            "lambda (heavy slots)": self.lambda_,
            "heavy entries used": len(self.heavy),
            "CMS width (w)": self.w,
            "CMS depth (d)": self.d,
            "CMS counters": self.w * self.d,
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

    hc = LambdaHCount(epsilon=0.01, delta=0.01, lambda_=64)
    for w in stream:
        hc.update(w)

    print("=== λ-HCount ===")
    info = hc.space_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFrequency estimates:")
    true_counts = {w: stream.count(w) for w in words}
    for word in words:
        est = hc.query(word)
        true = true_counts[word]
        print(f"  {word:12s}: estimated={est:5d}  true={true:5d}  error={est-true:+d}")
