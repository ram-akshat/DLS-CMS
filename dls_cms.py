"""
DLS-CMS: Decayed Lazy Sparse Count-Min Sketch
==============================================
Supports:
  - Insert (positive updates)
  - Delete (negative updates)
  - Query  (decayed frequency estimate)

Usage:
  python dls_cms.py              — run all three examples
  python dls_cms.py interactive  — live REPL mode
"""

import hashlib
import math
import sys
import random


# ═══════════════════════════════════════════════════════════════
# Internal: one sparse CMS grid (CMS⁺ or CMS⁻)
# ═══════════════════════════════════════════════════════════════

class _SparseCMS:
    def __init__(self, d: int, w: int, lam: float, tau: float, seed: int = 0):
        self.d    = d
        self.w    = w
        self.lam  = lam
        self.tau  = tau
        self.maps = [{} for _ in range(d)]          # maps[row][bucket] = (value, t)
        self.salts = [seed * 1000 + i for i in range(d)]

    def _bucket(self, row: int, key: str) -> int:
        raw = f"{self.salts[row]}:{key}".encode()
        return int(hashlib.md5(raw).hexdigest(), 16) % self.w

    def _decayed(self, value: float, stored_t: int, now_t: int) -> float:
        delta = now_t - stored_t
        return value * (self.lam ** delta) if delta > 0 else value

    def update(self, key: str, amount: float, t: int):
        for row in range(self.d):
            bucket = self._bucket(row, key)
            if bucket in self.maps[row]:
                sv, st = self.maps[row][bucket]
                current = self._decayed(sv, st, t)
            else:
                current = 0.0
            new_val = current + amount
            if new_val < self.tau:
                self.maps[row].pop(bucket, None)
            else:
                self.maps[row][bucket] = (new_val, t)

    def get_row_estimate(self, key: str, row: int, t: int) -> float:
        bucket = self._bucket(row, key)
        if bucket in self.maps[row]:
            sv, st = self.maps[row][bucket]
            return self._decayed(sv, st, t)
        return 0.0

    def active_count(self) -> int:
        return sum(len(m) for m in self.maps)

    def dense_row(self, row: int, t: int) -> list:
        """Return a dense list of length w with current decayed values (for grid display)."""
        result = [0.0] * self.w
        for bucket, (sv, st) in self.maps[row].items():
            result[bucket] = self._decayed(sv, st, t)
        return result


# ═══════════════════════════════════════════════════════════════
# Main DLS-CMS
# ═══════════════════════════════════════════════════════════════

class DLSCMS:
    def __init__(self, epsilon: float, delta: float, lam: float, seed: int = 42):
        if not (0 < epsilon < 1): raise ValueError("epsilon must be in (0,1)")
        if not (0 < delta   < 1): raise ValueError("delta must be in (0,1)")
        if not (0 < lam     < 1): raise ValueError("lam must be in (0,1)")

        self.epsilon = epsilon
        self.delta   = delta
        self.lam     = lam
        self.F       = 1.0 / (1.0 - lam)
        self.w       = math.ceil(math.e / epsilon)
        self.d       = math.ceil(math.log(1.0 / delta))
        self.tau     = epsilon * self.F / 2.0

        self.cms_plus  = _SparseCMS(self.d, self.w, lam, self.tau, seed=seed)
        self.cms_minus = _SparseCMS(self.d, self.w, lam, self.tau, seed=seed + 9999)

        self.current_time = 0
        self._seen_keys   = set()          # track all items ever inserted/deleted

    # ── Public API ──────────────────────────────────────────────

    def insert(self, key: str, amount: float = 1.0, t: int = None):
        if amount <= 0:
            raise ValueError("insert amount must be positive.")
        t = self._resolve_time(t)
        self._seen_keys.add(key)
        self.cms_plus.update(key, amount, t)

    def delete(self, key: str, amount: float = 1.0, t: int = None):
        if amount <= 0:
            raise ValueError("delete amount must be positive.")
        t = self._resolve_time(t)
        self._seen_keys.add(key)
        self.cms_minus.update(key, amount, t)

    def query(self, key: str, t: int = None) -> float:
        if t is None:
            t = self.current_time
        row_ests = []
        for row in range(self.d):
            vp = self.cms_plus.get_row_estimate(key, row, t)
            vm = self.cms_minus.get_row_estimate(key, row, t)
            row_ests.append(vp - vm)
        return max(0.0, min(row_ests))

    def query_all(self, t: int = None) -> dict:
        """Query every item that has been seen so far."""
        if t is None:
            t = self.current_time
        return {k: self.query(k, t) for k in sorted(self._seen_keys)}

    def tick(self, steps: int = 1):
        self.current_time += steps

    def info(self) -> dict:
        return {
            "rows (d)"          : self.d,
            "columns (w)"       : self.w,
            "lambda"            : self.lam,
            "epsilon"           : self.epsilon,
            "delta"             : self.delta,
            "F (total weight)"  : self.F,
            "pruning tau"       : self.tau,
            "current_time"      : self.current_time,
            "active_plus"       : self.cms_plus.active_count(),
            "active_minus"      : self.cms_minus.active_count(),
        }

    # ── Grid display ────────────────────────────────────────────

    def print_grid(self, label: str = "", max_cols: int = 30):
        """
        Print a compact CMS⁺ − CMS⁻ net grid, like SD-CMS's counter grid.
        Shows net = (plus_value - minus_value) per cell.
        Only active (non-zero) columns are shown; capped at max_cols wide.
        """
        t = self.current_time
        print()

        # Collect all active buckets across both sketches
        active_buckets = set()
        for row in range(self.d):
            active_buckets.update(self.cms_plus.maps[row].keys())
            active_buckets.update(self.cms_minus.maps[row].keys())

        cols = sorted(active_buckets)[:max_cols]

        if label:
            print(f"  Net Counter Grid (CMS⁺ − CMS⁻) at t={t}  [{label}]")
        else:
            print(f"  Net Counter Grid (CMS⁺ − CMS⁻) at t={t}")

        # Header
        header = "     " + "".join(f"  c{c:<5}" for c in cols)
        print(header)

        # Rows
        for row in range(self.d):
            plus_row  = self.cms_plus.dense_row(row, t)
            minus_row = self.cms_minus.dense_row(row, t)
            cells = []
            for c in cols:
                net = plus_row[c] - minus_row[c]
                if net >= 0:
                    cells.append(f"  +{net:.3f}")
                else:
                    cells.append(f"  {net:.3f}")
            print(f"r{row} [{''.join(cells)}  ]")
        print()

    # ── Internal ────────────────────────────────────────────────

    def _resolve_time(self, t):
        if t is None:
            self.current_time += 1
            return self.current_time
        self.current_time = max(self.current_time, t)
        return t
    
        # ── Space estimation (IMPORTANT for comparison) ─────────────
    def space_bytes(self) -> int:
        """
        Estimate memory usage of DLS-CMS.

        Each active entry stores:
            - value (float) → 8 bytes
            - timestamp (int) → 4 bytes
            - key (bucket index) → ~4 bytes
        Approx = 16 bytes per active counter
        """

        bytes_per_entry = 16

        total_entries = (
            self.cms_plus.active_count() +
            self.cms_minus.active_count()
        )

        return total_entries * bytes_per_entry


# ═══════════════════════════════════════════════════════════════
# Print init banner (matches SD-CMS style)
# ═══════════════════════════════════════════════════════════════

def _print_init(sketch: DLSCMS):
    print(f"DLS-CMS initialised:")
    print(f"  λ={sketch.lam},  ε={sketch.epsilon},  δ={sketch.delta}")
    print(f"  F = 1/(1-λ) = {sketch.F:.2f}")
    print(f"  CMS⁺ / CMS⁻ width = {sketch.w} columns × {sketch.d} rows")
    print(f"  Active counters    = {sketch.cms_plus.active_count()} (plus) / {sketch.cms_minus.active_count()} (minus)")
    print(f"  Pruning threshold  τ = {sketch.tau:.4f}")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 1: Basic Insert / Delete / Query  (mirrors your SD-CMS ex1)
# ═══════════════════════════════════════════════════════════════

def example1():
    print("=" * 60)
    print("EXAMPLE 1: Basic Insert / Delete / Query")
    print("=" * 60)

    sketch = DLSCMS(epsilon=0.05, delta=0.1, lam=0.9, seed=1)
    _print_init(sketch)
    print()

    ops = [
        ("INSERT", "A", 1), ("INSERT", "B", 2), ("INSERT", "A", 3),
        ("INSERT", "C", 4), ("INSERT", "A", 5), ("INSERT", "B", 6),
        ("DELETE", "A", 7),
    ]

    for op, item, t in ops:
        if op == "INSERT":
            sketch.insert(item, t=t)
        else:
            sketch.delete(item, t=t)
        print(f"t={t}: {op} {item}")

    sketch.print_grid()

    print(f"Queries at t={sketch.current_time + 1}:")
    for item in ["A", "B", "C", "D"]:
        est = sketch.query(item, t=sketch.current_time + 1)
        print(f"  freq({item}) ≈ {est:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 2: Network Packet Stream
# ═══════════════════════════════════════════════════════════════

def example2():
    print("=" * 60)
    print("EXAMPLE 2: Network Packet Stream (IP Frequency Estimation)")
    print("=" * 60)

    sketch = DLSCMS(epsilon=0.01, delta=0.05, lam=0.95, seed=42)
    _print_init(sketch)
    print()

    random.seed(99)
    base_ips   = [f"192.168.1.{i}" for i in range(1, 20)]
    weights    = [438, 40, 39, 38, 33, 20, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    true_counts = {}

    events = []
    for ip, w in zip(base_ips, weights):
        true_counts[ip] = w
        events.extend([ip] * w)
    random.shuffle(events)

    for t, ip in enumerate(events, start=1):
        sketch.insert(ip, t=t)

    # Delete some packets for the top IP (simulates flow expiry)
    for _ in range(10):
        sketch.delete("192.168.1.1")

    top5 = sorted(true_counts.items(), key=lambda x: -x[1])[:5]
    print("Top 5 IPs by TRUE count (no decay, for reference):")
    for ip, tc in top5:
        est = sketch.query(ip)
        print(f"  {ip:<20}  true={tc:>4}   estimated(decayed)={est:.4f}")

    print()
    print("Sketch Info:")
    for k, v in sketch.info().items():
        print(f"  {k:<24} : {v}")
    print()


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 3: Word Frequency in Streaming Text
# ═══════════════════════════════════════════════════════════════

def example3():
    print("=" * 60)
    print("EXAMPLE 3: Word Frequency in Streaming Text")
    print("=" * 60)

    sketch = DLSCMS(epsilon=0.05, delta=0.1, lam=0.85, seed=7)
    _print_init(sketch)
    print()

    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the fox ran quickly over the hill",
        "the dog barked at the fox",
        "the fox escaped over the hill",
        "the dog chased the fox and the fox ran",
        "the fox returned home",
        "the dog met the fox again",
    ]

    words = []
    for sentence in sentences:
        words.extend(sentence.split())

    true_counts = {}
    for w in words:
        true_counts[w] = true_counts.get(w, 0) + 1

    for t, word in enumerate(words, start=1):
        sketch.insert(word, t=t)

    print(f"Total words processed: {len(words)}")
    print(f"Unique words: {len(true_counts)}")
    print()
    print(f"{'Word':<15}  {'True Count':>10}    {'DLS-CMS Estimate':>16}")
    print("-" * 49)

    sorted_words = sorted(true_counts.items(), key=lambda x: -x[1])
    for word, tc in sorted_words:
        est = sketch.query(word)
        print(f"{word:<15}  {tc:>10}           {est:.4f}")

    print()
    print("─" * 60)
    print("Run with 'python dls_cms.py interactive' for live mode.")
    print("─" * 60)
    print()


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════

def interactive():
    print("=" * 60)
    print("INTERACTIVE MODE — DLS-CMS")
    print("=" * 60)
    print("Commands:")
    print("  insert <item>         — insert item at next timestamp")
    print("  delete <item>         — delete item")
    print("  query  <item>         — estimate decayed frequency")
    print("  queryall              — query all seen items")
    print("  info                  — show sketch parameters")
    print("  grid                  — print net counter grid")
    print("  tick [n]              — advance time by n steps (default 1)")
    print("  quit                  — exit")
    print()

    sketch = DLSCMS(epsilon=0.1, delta=0.1, lam=0.9, seed=0)
    _print_init(sketch)
    print()

    while True:
        try:
            raw = input("dls-cms> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd == "quit":
            print("Bye!")
            break

        elif cmd == "insert":
            if len(parts) < 2:
                print("  Usage: insert <item>")
                continue
            item = parts[1]
            sketch.insert(item)
            print(f"  t={sketch.current_time}: INSERT {item}  →  freq({item}) ≈ {sketch.query(item):.4f}")

        elif cmd == "delete":
            if len(parts) < 2:
                print("  Usage: delete <item>")
                continue
            item = parts[1]
            sketch.delete(item)
            print(f"  t={sketch.current_time}: DELETE {item}  →  freq({item}) ≈ {sketch.query(item):.4f}")

        elif cmd == "query":
            if len(parts) < 2:
                print("  Usage: query <item>")
                continue
            item = parts[1]
            est  = sketch.query(item)
            print(f"  freq({item}) ≈ {est:.4f}  (at t={sketch.current_time})")

        elif cmd == "queryall":
            results = sketch.query_all()
            if not results:
                print("  No items seen yet.")
            else:
                print(f"  {'Item':<20}  Estimate")
                print(f"  {'-'*20}  --------")
                for k, v in sorted(results.items(), key=lambda x: -x[1]):
                    print(f"  {k:<20}  {v:.4f}")

        elif cmd == "info":
            print()
            for k, v in sketch.info().items():
                print(f"  {k:<24} : {v}")
            print()

        elif cmd == "grid":
            sketch.print_grid()

        elif cmd == "tick":
            steps = int(parts[1]) if len(parts) > 1 else 1
            sketch.tick(steps)
            print(f"  Ticked {steps} step(s). Current time = {sketch.current_time}")

        else:
            print(f"  Unknown command: '{cmd}'. Type 'quit' to exit.")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive()
    else:
        example1()
        example2()
        example3()