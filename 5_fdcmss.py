"""
FDCMSS – Frequency-Decay Count-Min Sketch with Sliding window Support
Authors: Cafaro, Epicoco & Pulimeno (2016)
Paper: "Fast Data Stream Summaries"

FDCMSS augments Count-Min Sketch with:
  1. Frequency-decay: older items' counts decay exponentially (factor α per window)
  2. Sliding-window counters per cell (one per sub-window epoch)

Space: O(w * d * num_epochs) counters
"""

import math
import hashlib


class FDCMSS:
    def __init__(self, epsilon=0.01, delta=0.01, alpha=0.5, window_size=1000, num_epochs=4):
        """
        epsilon, delta : accuracy / failure probability
        alpha          : decay factor per epoch (0 < alpha < 1)
        window_size    : items per epoch
        num_epochs     : number of sliding-window sub-windows kept
        """
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.window_size = window_size
        self.num_epochs = num_epochs

        self.w = math.ceil(math.e / epsilon)
        self.d = math.ceil(math.log(1 / delta))

        # 3-D table: [depth][width][epoch]
        self.table = [[[0.0] * num_epochs for _ in range(self.w)] for _ in range(self.d)]

        self.current_epoch = 0
        self.items_in_epoch = 0
        self.n = 0

    # ── Hash ─────────────────────────────────────────────────────────────────

    def _hash(self, item, row):
        h = hashlib.md5(f"{item}_{row}".encode()).hexdigest()
        return int(h, 16) % self.w

    # ── Epoch management ─────────────────────────────────────────────────────

    def _advance_epoch(self):
        """Rotate epoch slot and apply decay to all previous epochs."""
        self.current_epoch = (self.current_epoch + 1) % self.num_epochs
        # Decay all epoch slots
        for i in range(self.d):
            for j in range(self.w):
                for e in range(self.num_epochs):
                    self.table[i][j][e] *= self.alpha
                # Clear the new current epoch slot (it was the oldest)
                self.table[i][j][self.current_epoch] = 0.0
        self.items_in_epoch = 0

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, item, count=1):
        self.n += count
        self.items_in_epoch += count
        if self.items_in_epoch >= self.window_size:
            self._advance_epoch()

        for i in range(self.d):
            col = self._hash(item, i)
            self.table[i][col][self.current_epoch] += count

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, item):
        """Sum across epochs (with decay already applied), take row-min."""
        estimates = []
        for i in range(self.d):
            col = self._hash(item, i)
            total = sum(self.table[i][col][e] for e in range(self.num_epochs))
            estimates.append(total)
        return int(min(estimates))

    # ── Space ─────────────────────────────────────────────────────────────────

    def space_bytes(self):
        # Each cell is a float64 (8 bytes)
        return self.w * self.d * self.num_epochs * 8

    def space_info(self):
        return {
            "algorithm": "FDCMSS",
            "width (w)": self.w,
            "depth (d)": self.d,
            "num_epochs": self.num_epochs,
            "alpha (decay)": self.alpha,
            "total_counters": self.w * self.d * self.num_epochs,
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

    fd = FDCMSS(epsilon=0.01, delta=0.01, alpha=0.5, window_size=500, num_epochs=4)
    for w in stream:
        fd.update(w)

    print("=== FDCMSS ===")
    info = fd.space_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFrequency estimates:")
    true_counts = {w: stream.count(w) for w in words}
    for word in words:
        est = fd.query(word)
        true = true_counts[word]
        print(f"  {word:12s}: estimated={est:5d}  true={true:5d}  error={est-true:+d}")
