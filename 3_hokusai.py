"""
Hokusai
Authors: Matusevych, Smola & Ahmed (2012)
Paper: "Hokusai – Sketching Streams in Real Time"

Hokusai extends Count-Min Sketch with time awareness.
It stores multiple CMS snapshots at different time resolutions
using a logarithmic aging strategy (halving resolution each level).

Space: O(w * d * log(T)) where T = number of time ticks observed.
Here we simulate it with a fixed number of levels for a stream of known length.
"""

import math
import hashlib


class HokusaiCMS:
    """Single CMS layer used inside Hokusai."""

    def __init__(self, w, d, seed=0):
        self.w = w
        self.d = d
        self.seed = seed
        self.table = [[0] * w for _ in range(d)]

    def _hash(self, item, row):
        h = hashlib.md5(f"{self.seed}_{item}_{row}".encode()).hexdigest()
        return int(h, 16) % self.w

    def update(self, item, count=1):
        for i in range(self.d):
            self.table[i][self._hash(item, i)] += count

    def query(self, item):
        return min(self.table[i][self._hash(item, i)] for i in range(self.d))

    def merge_into(self, other):
        """Merge this sketch into another (for aging / compaction)."""
        for i in range(self.d):
            for j in range(self.w):
                other.table[i][j] += self.table[i][j]

    def reset(self):
        self.table = [[0] * self.w for _ in range(self.d)]

    def counter_count(self):
        return self.w * self.d


class Hokusai:
    """
    Simplified Hokusai with L levels.
    Level 0 = finest granularity (one CMS per tick).
    Each subsequent level covers 2x the time span, achieved by merging pairs.
    """

    def __init__(self, epsilon=0.01, delta=0.01, levels=4):
        self.epsilon = epsilon
        self.delta = delta
        self.levels = levels
        self.w = math.ceil(math.e / epsilon)
        self.d = math.ceil(math.log(1 / delta))

        # One active CMS per level
        self.active = [HokusaiCMS(self.w, self.d, seed=lvl) for lvl in range(levels)]
        # Archive: list of completed sketches per level
        self.archive = [[] for _ in range(levels)]
        self.tick = 0
        self.total_items = 0

    def _advance_tick(self):
        """Propagate carry (like binary counting) across levels."""
        self.tick += 1
        carry = True
        for lvl in range(self.levels):
            if carry:
                self.archive[lvl].append(self.active[lvl])
                self.active[lvl] = HokusaiCMS(self.w, self.d, seed=lvl)
                # Compact: merge every 2 archived sketches at this level into next
                if len(self.archive[lvl]) >= 2:
                    merged = HokusaiCMS(self.w, self.d, seed=lvl + 1)
                    self.archive[lvl][-2].merge_into(merged)
                    self.archive[lvl][-1].merge_into(merged)
                    self.archive[lvl] = self.archive[lvl][:-2]
                    # Push merged into next level's archive
                    if lvl + 1 < self.levels:
                        self.archive[lvl + 1].append(merged)
                    carry = True
                else:
                    carry = False
            else:
                break

    def update(self, item, count=1, advance_tick=False):
        self.total_items += count
        self.active[0].update(item, count)
        if advance_tick:
            self._advance_tick()

    def query(self, item):
        """Point query: min across all active sketches."""
        estimates = [self.active[lvl].query(item) for lvl in range(self.levels)]
        # Also check archives
        for lvl in range(self.levels):
            for sk in self.archive[lvl]:
                estimates.append(sk.query(item))
        return min(estimates) if estimates else 0

    def total_counters(self):
        cnt = sum(a.counter_count() for a in self.active)
        cnt += sum(sk.counter_count() for lvl in self.archive for sk in lvl)
        return cnt

    def space_bytes(self):
        return self.total_counters() * 4

    def space_info(self):
        return {
            "algorithm": "Hokusai",
            "levels": self.levels,
            "width (w)": self.w,
            "depth (d)": self.d,
            "total_counters": self.total_counters(),
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

    hok = Hokusai(epsilon=0.01, delta=0.01, levels=4)
    for idx, w in enumerate(stream):
        advance = (idx % 100 == 99)  # new tick every 100 items
        hok.update(w, advance_tick=advance)

    print("=== Hokusai ===")
    info = hok.space_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFrequency estimates:")
    true_counts = {w: stream.count(w) for w in words}
    for word in words:
        est = hok.query(word)
        true = true_counts[word]
        print(f"  {word:12s}: estimated={est:5d}  true={true:5d}  error={est-true:+d}")
