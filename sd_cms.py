"""
SD-CMS: Signed Decay Count-Min Sketch
======================================
Supports: Insert, Delete, Query
Model   : Exponential time decay (sliding window approximation)
Estimator: Median of signed counters (unbiased)

Author  : [Your Name]
"""

import hashlib
import math
import statistics
import time
from typing import Union


# ─────────────────────────────────────────────
#  Core SD-CMS Class
# ─────────────────────────────────────────────

class SDCMS:
    """
    Signed Decay Count-Min Sketch.

    Parameters
    ----------
    epsilon : float
        Acceptable relative error (e.g. 0.01 = 1%).
        Controls width w = ceil(e * F / epsilon).

    delta : float
        Failure probability (e.g. 0.05 = 5% chance of exceeding error).
        Controls depth d = ceil(ln(1/delta)).

    lambda_ : float
        Decay factor per time unit. Must be in (0, 1).
        e.g. 0.9 means events 10 steps ago have weight 0.9^10 ≈ 0.35

    How sizes are derived
    ---------------------
    F = 1 / (1 - lambda_)          # converged total weight (geometric series)
    w = ceil(e * F / epsilon)      # number of columns
    d = ceil(ln(1 / delta))        # number of rows (hash functions)
    """

    def __init__(self, epsilon: float = 0.01, delta: float = 0.05, lambda_: float = 0.9):
        if not (0 < lambda_ < 1):
            raise ValueError("lambda_ must be strictly between 0 and 1")
        if epsilon <= 0 or delta <= 0:
            raise ValueError("epsilon and delta must be positive")

        self.epsilon  = epsilon
        self.delta    = delta
        self.lambda_  = lambda_

        # Converged total decayed weight
        self.F = 1.0 / (1.0 - lambda_)

        # Sketch dimensions
        self.w = math.ceil(math.e * self.F / epsilon)   # columns
        self.d = math.ceil(math.log(1.0 / delta))       # rows

        # 2D counter grid [d rows x w columns], all zeros
        self.C = [[0.0] * self.w for _ in range(self.d)]

        # Time tracking
        self.current_time = 0
        self.last_decay_time = 0

        # Hash seeds — one per row, for both position hash and sign hash
        self.hash_seeds_pos  = [i * 2654435761 for i in range(1, self.d + 1)]
        self.hash_seeds_sign = [i * 2246822519 for i in range(1, self.d + 1)]

        print(f"SD-CMS initialised:")
        print(f"  λ={lambda_},  ε={epsilon},  δ={delta}")
        print(f"  F = 1/(1-λ) = {self.F:.2f}")
        print(f"  Grid size   = {self.d} rows × {self.w} columns")
        print(f"  Total cells = {self.d * self.w}\n")

    # ── Internal Hashing ──────────────────────────────────────────────

    def _hash_position(self, item: str, row: int) -> int:
        """Maps item to a column index in [0, w-1] for a given row."""
        seed = self.hash_seeds_pos[row]
        raw  = int(hashlib.md5(f"{seed}:{item}".encode()).hexdigest(), 16)
        return raw % self.w

    def _hash_sign(self, item: str, row: int) -> int:
        """Returns +1 or -1 for item in a given row."""
        seed = self.hash_seeds_sign[row]
        raw  = int(hashlib.md5(f"{seed}:{item}".encode()).hexdigest(), 16)
        return 1 if (raw % 2 == 0) else -1

    # ── Decay Application ─────────────────────────────────────────────

    def _apply_decay(self, new_time: int):
        """
        Multiply every counter by λ^(new_time - last_decay_time).
        This is the O(d·w) step executed on every update.
        """
        elapsed = new_time - self.last_decay_time
        if elapsed <= 0:
            return

        alpha = self.lambda_ ** elapsed

        for i in range(self.d):
            for j in range(self.w):
                self.C[i][j] *= alpha

        self.last_decay_time = new_time
        self.current_time    = new_time

    # ── Public API ────────────────────────────────────────────────────

    def update(self, item: str, delta: float = 1.0, timestamp: int = None):
        """
        Insert (delta=+1) or Delete (delta=-1) an item at a given timestamp.

        Parameters
        ----------
        item      : The item identifier (any string, e.g. IP address, word).
        delta     : +1 for insert, -1 for delete. Can also be a weight.
        timestamp : Integer time step. If None, auto-increments by 1.

        Time Complexity: O(d·w) — dominated by full-matrix decay.
        """
        if timestamp is None:
            timestamp = self.current_time + 1

        # Step 1: Decay all counters to current time
        self._apply_decay(timestamp)

        # Step 2: Update d cells — one per row
        for i in range(self.d):
            col  = self._hash_position(item, i)
            sign = self._hash_sign(item, i)
            self.C[i][col] += sign * delta

    def insert(self, item: str, timestamp: int = None):
        """Insert item into the sketch. Calls update(delta=+1)."""
        self.update(item, delta=+1.0, timestamp=timestamp)

    def delete(self, item: str, timestamp: int = None):
        """Delete item from the sketch. Calls update(delta=-1)."""
        self.update(item, delta=-1.0, timestamp=timestamp)

    def query(self, item: str) -> float:
        """
        Estimate the current decayed frequency of item.

        Returns the median of s_i(item) * C[i][h_i(item)] across all rows.
        Median gives an unbiased estimate (unlike CMS min which always overestimates).

        Time Complexity: O(d) — only touches one cell per row.
        """
        estimates = []
        for i in range(self.d):
            col  = self._hash_position(item, i)
            sign = self._hash_sign(item, i)
            estimates.append(sign * self.C[i][col])

        return statistics.median(estimates)

    def query_all(self, items: list) -> dict:
        """Query multiple items at once. Returns {item: estimated_frequency}."""
        return {item: self.query(item) for item in items}

    # ── Utilities ─────────────────────────────────────────────────────

    def advance_time(self, new_time: int):
        """
        Advance time without any update — just applies decay.
        Useful when time passes but no events arrive.
        """
        self._apply_decay(new_time)

    def get_sketch_info(self) -> dict:
        """Returns current sketch metadata."""
        return {
            "rows (d)"           : self.d,
            "columns (w)"        : self.w,
            "lambda"             : self.lambda_,
            "epsilon"            : self.epsilon,
            "delta"              : self.delta,
            "F (total weight)"   : round(self.F, 4),
            "current_time"       : self.current_time,
            "total_cells"        : self.d * self.w,
        }

    def print_grid(self, precision: int = 3):
        """Print the current counter grid (for debugging small sketches)."""
        print(f"\nCounter Grid at t={self.current_time}:")
        print("     " + "  ".join(f"c{j:<4}" for j in range(self.w)))
        for i, row in enumerate(self.C):
            vals = "  ".join(f"{v:+.{precision}f}" for v in row)
            print(f"r{i} [  {vals}  ]")
        print()


# ─────────────────────────────────────────────
#  Example 1: Basic Walkthrough
# ─────────────────────────────────────────────

def example_basic():
    print("=" * 60)
    print("EXAMPLE 1: Basic Insert / Delete / Query")
    print("=" * 60)

    # Small sketch so grid is printable
    sketch = SDCMS(epsilon=0.5, delta=0.1, lambda_=0.9)

    # Insert sequence
    events = [
        (1,  "A", "insert"),
        (2,  "B", "insert"),
        (3,  "A", "insert"),
        (4,  "C", "insert"),
        (5,  "A", "insert"),
        (6,  "B", "insert"),
        (7,  "A", "delete"),   # ← delete one occurrence of A
    ]

    for t, item, op in events:
        if op == "insert":
            sketch.insert(item, timestamp=t)
            print(f"t={t}: INSERT {item}")
        else:
            sketch.delete(item, timestamp=t)
            print(f"t={t}: DELETE {item}")

    sketch.print_grid()

    # Query at t=8
    print("Queries at t=8:")
    for item in ["A", "B", "C", "D"]:
        freq = sketch.query(item)
        print(f"  freq({item}) ≈ {freq:.4f}")

    print()


# ─────────────────────────────────────────────
#  Example 2: Network Packet Stream Simulation
# ─────────────────────────────────────────────

def example_network():
    print("=" * 60)
    print("EXAMPLE 2: Network Packet Stream (IP Frequency Estimation)")
    print("=" * 60)

    sketch = SDCMS(epsilon=0.01, delta=0.05, lambda_=0.95)

    # Simulate 1000 packets
    import random
    random.seed(42)

    ip_pool = [f"192.168.1.{i}" for i in range(1, 21)]  # 20 IPs
    # IP .1 is a heavy hitter — appears 40% of the time
    heavy_hitter = "192.168.1.1"

    true_counts  = {}
    total_events = 1000

    for t in range(1, total_events + 1):
        # Biased sampling: heavy hitter more likely
        if random.random() < 0.40:
            ip = heavy_hitter
        else:
            ip = random.choice(ip_pool)

        sketch.insert(ip, timestamp=t)
        true_counts[ip] = true_counts.get(ip, 0) + 1

    # Compare estimated vs true (using decayed true counts at final time)
    # True decayed count = sum of λ^(T-t) for each occurrence at time t
    # We approximate here just by showing top IPs

    print(f"\nTop 5 IPs by TRUE count (no decay, for reference):")
    sorted_true = sorted(true_counts.items(), key=lambda x: -x[1])
    for ip, cnt in sorted_true[:5]:
        est = sketch.query(ip)
        print(f"  {ip:<18}  true={cnt:4d}   estimated(decayed)={est:.4f}")

    print()
    info = sketch.get_sketch_info()
    print("Sketch Info:")
    for k, v in info.items():
        print(f"  {k:<25}: {v}")
    print()


# ─────────────────────────────────────────────
#  Example 3: Word Frequency in a Text Stream
# ─────────────────────────────────────────────

def example_text_stream():
    print("=" * 60)
    print("EXAMPLE 3: Word Frequency in Streaming Text")
    print("=" * 60)

    sketch = SDCMS(epsilon=0.05, delta=0.1, lambda_=0.85)

    sentence = (
        "the quick brown fox jumps over the lazy dog "
        "the fox ran quickly over the hill and the dog barked "
        "the quick fox and the lazy dog met again "
        "over the hill the dog chased the fox "
        "the fox escaped and the dog returned home"
    )

    words = sentence.split()
    true_counts = {}

    for t, word in enumerate(words, start=1):
        sketch.insert(word, timestamp=t)
        true_counts[word] = true_counts.get(word, 0) + 1

    print(f"\nTotal words processed: {len(words)}")
    print(f"Unique words: {len(true_counts)}\n")

    print(f"{'Word':<12} {'True Count':>12} {'SD-CMS Estimate':>18}")
    print("-" * 45)

    for word in sorted(true_counts, key=lambda w: -true_counts[w]):
        est = sketch.query(word)
        print(f"{word:<12} {true_counts[word]:>12}     {est:>12.4f}")

    print()


# ─────────────────────────────────────────────
#  Example 4: Interactive Command-Line Mode
# ─────────────────────────────────────────────

def interactive_mode():
    print("=" * 60)
    print("INTERACTIVE MODE — SD-CMS")
    print("=" * 60)
    print("Commands:")
    print("  insert <item>         — insert item at next timestamp")
    print("  delete <item>         — delete item")
    print("  query  <item>         — estimate decayed frequency")
    print("  queryall              — query all seen items")
    print("  info                  — show sketch parameters")
    print("  grid                  — print counter grid")
    print("  quit                  — exit")
    print()

    # Default parameters — user can change these before running
    sketch = SDCMS(epsilon=0.1, delta=0.1, lambda_=0.9)
    seen_items = set()
    t = 0

    while True:
        try:
            raw = input("sd-cms> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd == "quit":
            break

        elif cmd == "insert" and len(parts) == 2:
            t += 1
            item = parts[1]
            sketch.insert(item, timestamp=t)
            seen_items.add(item)
            print(f"  Inserted '{item}' at t={t}")

        elif cmd == "delete" and len(parts) == 2:
            t += 1
            item = parts[1]
            sketch.delete(item, timestamp=t)
            print(f"  Deleted '{item}' at t={t}")

        elif cmd == "query" and len(parts) == 2:
            item = parts[1]
            freq = sketch.query(item)
            print(f"  freq('{item}') ≈ {freq:.6f}")

        elif cmd == "queryall":
            if not seen_items:
                print("  No items inserted yet.")
            else:
                results = sketch.query_all(list(seen_items))
                for item, freq in sorted(results.items(), key=lambda x: -x[1]):
                    print(f"  {item:<20} {freq:.6f}")

        elif cmd == "info":
            for k, v in sketch.get_sketch_info().items():
                print(f"  {k:<25}: {v}")

        elif cmd == "grid":
            sketch.print_grid()

        else:
            print("  Unknown command. Try: insert <item>, delete <item>, query <item>, info, grid, quit")


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        example_basic()
        example_network()
        example_text_stream()
        print("─" * 60)
        print("Run with 'python sd_cms.py interactive' for live mode.")
        print("─" * 60)