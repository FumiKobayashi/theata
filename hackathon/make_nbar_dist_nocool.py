# -*- coding: utf-8 -*-
"""Gate-time phonon-number distribution under no cooling (d=3, d=5).

Soft-decoder hackathon material: the per-gate n-bar values that feed the
MS-gate error lookup (max transversal n-bar of the gate pair, 3 rounds,
deterministic schedule -> identical across shots).

Outputs (this directory):
  nbar_dist_nocool.png   two-panel histogram, bin width 0.25 (= lookup grid)
  nbar_gates_nocool.csv  per-gate raw values: d, round, gate_index, n_bar
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent
for sub in ("src", "circuits", "experiments"):
    sys.path.insert(0, str(REPO / sub))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt

NAVY = "#1F5575"
TEXT = "#202020"
MUTED = "#808080"
GRID = "#DCDCDC"
BIN = 0.25  # = lookup-table grid step


def compile_rounds(d, rounds=3):
    forward, reverse, _ = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    placement = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)
    cfg = CompilerConfig()
    results, zone_contents = [], None
    for r in range(1, rounds + 1):
        name = forward if r % 2 == 1 else reverse
        m = __import__(name)
        dag = DAGBuilder(parse(m.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=cfg).compile(dag, placement)
        results.append(cr)
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
    return results


def gate_nbars(results):
    rows = []  # (round, gate_index, n_bar)
    for rnd, cr in enumerate(results, start=1):
        for gi, g in enumerate((cr.phonon_data or {}).get("at_gates", [])):
            rows.append((rnd, gi, max(p[1] for p in g["phonons"].values())))
    return rows


data = {d: gate_nbars(compile_rounds(d)) for d in (3, 5)}

with open(HERE / "nbar_gates_nocool.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["d", "round", "gate_index", "n_bar_max_transversal"])
    for d, rows in data.items():
        for rnd, gi, v in rows:
            w.writerow([d, rnd, gi, f"{v:.4f}"])

plt.rcParams["font.family"] = "Segoe UI"
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=200)

for ax, d in zip(axes, (3, 5)):
    vals = [v for _, _, v in data[d]]
    mean = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    hi = (int(max(vals) / BIN) + 2) * BIN
    bins = np.arange(0, hi + BIN, BIN)
    ax.hist(vals, bins=bins, color=NAVY, edgecolor="white", linewidth=0.6,
            zorder=3)
    ax.axvline(mean, color=MUTED, ls="--", lw=1.2, zorder=4)
    ax.text(mean, ax.get_ylim()[1] * 0.97 if False else 0, "", va="bottom")
    ax.set_title(f"d = {d}   ({len(vals)} MS gates)", fontsize=14,
                 color=TEXT, loc="left")
    ax.annotate(f"mean {mean:.2f}\nsd {sd:.2f}",
                xy=(0.97, 0.92), xycoords="axes fraction",
                ha="right", va="top", fontsize=12, color=TEXT)
    ax.set_xlabel("gate-time n̄  (max transversal of the gate pair)",
                  fontsize=12, color=TEXT)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=11, colors=TEXT)

axes[0].set_ylabel("MS gates per 0.25-wide bin", fontsize=12, color=TEXT)
fig.suptitle("Gate-time phonon number, no cooling (3 rounds; "
             "deterministic schedule — identical every shot)",
             fontsize=13, color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "nbar_dist_nocool.png"
fig.savefig(out)
print("saved:", out)
print("saved:", HERE / "nbar_gates_nocool.csv")
