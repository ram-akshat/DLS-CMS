"""
Space Comparison – All 6 Algorithms
=====================================
Runs all algorithms on the SAME stream, collects space usage,
and produces two plots:

  1. Bar chart  – space (KB) at a fixed epsilon/delta
  2. Line chart – space (KB) vs epsilon  (delta fixed at 0.01)

Requirements:
    pip install matplotlib numpy

Run:
    python 6_compare_space.py
"""

import sys, os, importlib, random, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ── Make sure sibling files are importable ──────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from importlib.util import spec_from_file_location, module_from_spec

def load_class(filename, classname):
    path = os.path.join(HERE, filename)
    spec = spec_from_file_location(classname, path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)

CountMinSketch  = load_class("1_count_min_sketch.py", "CountMinSketch")
CountSketch     = load_class("2_count_sketch.py",     "CountSketch")
Hokusai         = load_class("3_hokusai.py",          "Hokusai")
LambdaHCount    = load_class("4_lambda_hcount.py",    "LambdaHCount")
FDCMSS          = load_class("5_fdcmss.py",           "FDCMSS")
SDCMS           = load_class("6_sd_cms.py",           "SDCMS")
DLSCMS = load_class("dls_cms.py", "DLSCMS")

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "Count-Min Sketch" : "#4C72B0",
    "Count Sketch"     : "#DD8452",
    "Hokusai"          : "#55A868",
    "λ-HCount"         : "#C44E52",
    "FDCMSS"           : "#8172B2",
    "SD_CMS"           : "#E63946",   # your algorithm – highlighted
    "DLS_CMS": "#2A9D8F",
}

# ── Common stream ─────────────────────────────────────────────────────────────
STREAM_SIZE = 10_000
random.seed(42)
VOCAB = ["apple","banana","cherry","date","elderberry",
         "fig","grape","honeydew","kiwi","lemon",
         "mango","nectarine","orange","papaya","quince"]

def generate_stream(n):
    for _ in range(n):
        yield random.choice(VOCAB)

true_counts = {}

# ── Helper: build + populate one instance ────────────────────────────────────

def run_cms(eps, dlt):
    obj = CountMinSketch(epsilon=eps, delta=dlt)
    true_counts = {}

    for w in generate_stream(STREAM_SIZE):
        obj.update(w)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_cs(eps, dlt):
    obj = CountSketch(epsilon=eps, delta=dlt)
    true_counts = {}

    for w in generate_stream(STREAM_SIZE):
        obj.update(w)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_hokusai(eps, dlt):
    obj = Hokusai(epsilon=eps, delta=dlt, levels=4)
    true_counts = {}

    for idx, w in enumerate(generate_stream(STREAM_SIZE)):
        obj.update(w, advance_tick=(idx % 100 == 99))
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_hcount(eps, dlt):
    obj = LambdaHCount(epsilon=eps, delta=dlt, lambda_=64)
    true_counts = {}

    for w in generate_stream(STREAM_SIZE):
        obj.update(w)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_fdcmss(eps, dlt):
    obj = FDCMSS(epsilon=eps, delta=dlt, alpha=0.5, window_size=500, num_epochs=4)
    true_counts = {}

    for w in generate_stream(STREAM_SIZE):
        obj.update(w)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_sdcms(eps, dlt):
    obj = SDCMS(epsilon=eps, delta=dlt)
    true_counts = {}

    for w in generate_stream(STREAM_SIZE):
        obj.update(w)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

def run_dlscms(eps, dlt):
    obj = DLSCMS(epsilon=eps, delta=dlt, lam=0.95)
    true_counts = {}

    for i, w in enumerate(generate_stream(STREAM_SIZE)):
        obj.insert(w, t=i+1)
        true_counts[w] = true_counts.get(w, 0) + 1

    return obj, true_counts

RUNNERS = [
    ("Count-Min Sketch", run_cms),
    ("Count Sketch",     run_cs),
    ("Hokusai",          run_hokusai),
    ("λ-HCount",         run_hcount),
    ("FDCMSS",           run_fdcmss),
    ("SD_CMS",           run_sdcms),
    ("DLS_CMS",          run_dlscms),
]

def compute_mae(sketch, true_counts):
    errors = []
    for key in true_counts:
        est = sketch.query(key)
        true = true_counts[key]
        errors.append(abs(est - true))
    return np.mean(errors)


def compute_relative_error(sketch, true_counts):
    errors = []
    for key in true_counts:
        est = sketch.query(key)
        true = true_counts[key]
        errors.append(abs(est - true) / (true + 1e-9))
    return np.mean(errors)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Bar chart at fixed epsilon=0.01, delta=0.01
# ═════════════════════════════════════════════════════════════════════════════

EPS_FIXED = 0.01
DLT_FIXED = 0.01

print("Running all algorithms (fixed ε=0.01, δ=0.01) …")
bar_names, bar_kb = [], []
for name, runner in RUNNERS:
    obj, _ = runner(EPS_FIXED, DLT_FIXED)
    kb  = obj.space_bytes() / 1024
    bar_names.append(name)
    bar_kb.append(kb)
    print(f"  {name:20s}: {kb:.3f} KB")

fig1, ax1 = plt.subplots(figsize=(10, 6))
bar_colors = [COLORS[n] for n in bar_names]
bars = ax1.bar(bar_names, bar_kb, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6)

# Annotate bars
for bar, val in zip(bars, bar_kb):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(bar_kb) * 0.01,
             f"{val:.2f} KB",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1.set_title("Space Consumption Comparison\n(ε = 0.01, δ = 0.01, stream = 10 000 items)",
              fontsize=13, fontweight="bold", pad=14)
ax1.set_ylabel("Space (KB)", fontsize=11)
ax1.set_xlabel("Algorithm", fontsize=11)
ax1.tick_params(axis="x", labelsize=9)
ax1.set_ylim(0, max(bar_kb) * 1.2)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
ax1.set_axisbelow(True)

# Highlight SD_CMS bar
sd_idx = bar_names.index("SD_CMS")
bars[sd_idx].set_edgecolor("#333333")
bars[sd_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig("plot1_bar_space.png", dpi=150)
print("\n  → Saved: plot1_bar_space.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Line chart: Space vs epsilon  (delta fixed)
# ═════════════════════════════════════════════════════════════════════════════

EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

print("\nRunning all algorithms across ε values …")
line_data = {name: [] for name, _ in RUNNERS}

for eps in EPSILONS:
    for name, runner in RUNNERS:
        try:
            obj, _ = runner(eps, DLT_FIXED)
            kb  = obj.space_bytes() / 1024
        except Exception:
            kb = float("nan")
        line_data[name].append(kb)
    print(f"  ε={eps:.3f} done")

fig2, ax2 = plt.subplots(figsize=(10, 6))
for name, _ in RUNNERS:
    lw   = 2.8 if name == "SD_CMS" else 1.8
    ms   = 7   if name == "SD_CMS" else 5
    zord = 5   if name == "SD_CMS" else 2
    ax2.plot(EPSILONS, line_data[name],
             marker="o", label=name, color=COLORS[name],
             linewidth=lw, markersize=ms, zorder=zord)

ax2.set_title("Space (KB) vs Error Parameter ε\n(δ = 0.01, stream = 10 000 items)",
              fontsize=13, fontweight="bold", pad=14)
ax2.set_ylabel("Space (KB)", fontsize=11)
ax2.set_xlabel("ε (epsilon)", fontsize=11)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend(fontsize=9, framealpha=0.9)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.xaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("plot2_line_space_vs_epsilon.png", dpi=150)
print("  → Saved: plot2_line_space_vs_epsilon.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Space vs delta  (epsilon fixed)
# ═════════════════════════════════════════════════════════════════════════════

DELTAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

print("\nRunning all algorithms across δ values …")
delta_data = {name: [] for name, _ in RUNNERS}

for dlt in DELTAS:
    for name, runner in RUNNERS:
        try:
            obj, _ = runner(EPS_FIXED, dlt)
            kb  = obj.space_bytes() / 1024
        except Exception:
            kb = float("nan")
        delta_data[name].append(kb)
    print(f"  δ={dlt:.3f} done")

fig3, ax3 = plt.subplots(figsize=(10, 6))
for name, _ in RUNNERS:
    lw   = 2.8 if name == "SD_CMS" else 1.8
    ms   = 7   if name == "SD_CMS" else 5
    zord = 5   if name == "SD_CMS" else 2
    ax3.plot(DELTAS, delta_data[name],
             marker="s", label=name, color=COLORS[name],
             linewidth=lw, markersize=ms, zorder=zord)

ax3.set_title("Space (KB) vs Failure Probability δ\n(ε = 0.01, stream = 10 000 items)",
              fontsize=13, fontweight="bold", pad=14)
ax3.set_ylabel("Space (KB)", fontsize=11)
ax3.set_xlabel("δ (delta)", fontsize=11)
ax3.legend(fontsize=9, framealpha=0.9)
ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig("plot3_line_space_vs_delta.png", dpi=150)
print("  → Saved: plot3_line_space_vs_delta.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4 – Time Taken Comparison (Bar Chart)
# ═════════════════════════════════════════════════════════════════════════════

print("\nMeasuring execution time for all algorithms...")

time_names = []
time_vals = []

for name, runner in RUNNERS:
    start = time.time()
    obj, _ = runner(EPS_FIXED, DLT_FIXED)
    end = time.time()

    elapsed = end - start

    time_names.append(name)
    time_vals.append(elapsed)

    print(f"  {name:20s}: {elapsed:.6f} sec")

# Plot
fig4, ax4 = plt.subplots(figsize=(10, 6))
time_colors = [COLORS[n] for n in time_names]

bars = ax4.bar(time_names, time_vals, color=time_colors,
               edgecolor="white", linewidth=0.8, width=0.6)

# Annotate
for bar, val in zip(bars, time_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(time_vals) * 0.01,
             f"{val:.4f}s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

ax4.set_title("Execution Time Comparison\n(ε = 0.01, δ = 0.01, stream = 10 000 items)",
              fontsize=13, fontweight="bold", pad=14)
ax4.set_ylabel("Time (seconds)", fontsize=11)
ax4.set_xlabel("Algorithm", fontsize=11)
ax4.tick_params(axis="x", labelsize=9)
ax4.set_ylim(0, max(time_vals) * 1.3)
ax4.yaxis.grid(True, linestyle="--", alpha=0.5)
ax4.set_axisbelow(True)

# Highlight SD-CMS and DLS-CMS
if "SD_CMS" in time_names:
    idx = time_names.index("SD_CMS")
    bars[idx].set_edgecolor("#000000")
    bars[idx].set_linewidth(2.5)

if "DLS_CMS" in time_names:
    idx = time_names.index("DLS_CMS")
    bars[idx].set_edgecolor("#000000")
    bars[idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig("plot4_time_comparison.png", dpi=150)
print("  → Saved: plot4_time_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Accuracy Comparison (MAE)
# ═════════════════════════════════════════════════════════════════════════════

print("\nMeasuring accuracy (MAE)...")

acc_names = []
mae_vals = []
rel_vals = []

for name, runner in RUNNERS:
    try:
        obj, true_counts = runner(EPS_FIXED, DLT_FIXED)

        mae = compute_mae(obj, true_counts)
        rel = compute_relative_error(obj, true_counts)

    except Exception:
        mae = float("nan")
        rel = float("nan")

    acc_names.append(name)
    mae_vals.append(mae)
    rel_vals.append(rel)

    print(f"  {name:20s}: MAE={mae:.4f}, RelErr={rel:.4f}")

# Plot MAE
fig5, ax5 = plt.subplots(figsize=(10, 6))
colors = [COLORS[n] for n in acc_names]

bars = ax5.bar(acc_names, mae_vals, color=colors,
               edgecolor="white", linewidth=0.8, width=0.6)

for bar, val in zip(bars, mae_vals):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(mae_vals) * 0.01,
             f"{val:.3f}",
             ha="center", va="bottom", fontsize=9)

ax5.set_title("Mean Absolute Error (MAE) Comparison",
              fontsize=13, fontweight="bold")
ax5.set_ylabel("MAE", fontsize=11)
ax5.set_xlabel("Algorithm", fontsize=11)
ax5.yaxis.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot5_accuracy_mae.png", dpi=150)
print("  → Saved: plot5_accuracy_mae.png")


# ═════════════════════════════════════════════════════════════════════════════
# Show all plots
# ═════════════════════════════════════════════════════════════════════════════

print("\nAll plots saved. Displaying …")
plt.show()
