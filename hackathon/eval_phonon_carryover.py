# -*- coding: utf-8 -*-
"""Round-boundary phonon reset vs carry-over: does the paper's reset matter?

For each cooling setting (no cooling / theta=1.5), compile R rounds twice --
with the paper's round-boundary phonon reset and with per-ion carry-over
(previous round's final phonons seed the next round) -- and compare:
per-round gate-time n_bar trajectory, cooling count, wall time, and the
logical error rate of the whole R-round memory.

Hypothesis: under the threshold policy the carry-over is absorbed by the
first trigger checks (steady state), so LER barely moves; without cooling
n_bar accumulates across rounds and LER degrades toward 0.5.

Usage: venv\\Scripts\\python.exe hackathon\\eval_phonon_carryover.py --d 3
Results: hackathon/results_carryover_d{d}.csv (+ _rounds.csv per-round stats)
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_test_circuits import compile_rounds, build_noisy, LOOKUP  # noqa: E402,F401

from stim_recompiler import recompile_multi_round  # noqa: E402
from ion_timeline import gaps_from_compile_results  # noqa: E402
from exp_noise_injection import default_noise_params  # noqa: E402
from parallel_estimator import ParallelEstimator  # noqa: E402
import pymatching  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--shots", type=int, default=100_000)
args = ap.parse_args()
D, ROUNDS, SHOTS = args.d, args.rounds, args.shots
N_MAX = LOOKUP.n_bar_range[1]


def matcher_from(c):
    dem = c.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


round_rows, summary_rows = [], []
for theta in (None, 1.5):
    label = "no cooling" if theta is None else f"theta={theta:g}"
    for carry in (False, True):
        mode = "carry" if carry else "reset"
        t0 = time.perf_counter()
        results, observable = compile_rounds(D, ROUNDS, theta,
                                             carry_phonons=carry)
        circuit = recompile_multi_round(results, observable_column=observable)
        anchor_ops, gaps = gaps_from_compile_results(results)
        params = default_noise_params()

        at_gates, cools_total, clipped = [], 0, 0
        for rnd, cr in enumerate(results, start=1):
            ag = (cr.phonon_data or {}).get("at_gates", [])
            trs = [max(p[1] for p in g["phonons"].values()) for g in ag]
            cools = cr.stats.get("cools", 0)
            cools_total += cools
            clipped += sum(1 for v in trs if v > N_MAX)
            round_rows.append(dict(
                config=label, mode=mode, round=rnd, n_gates=len(trs),
                mean_nbar=f"{statistics.mean(trs):.3f}",
                max_nbar=f"{max(trs):.3f}", cools=cools))
            at_gates.extend(ag)

        pe = ParallelEstimator()
        t_ms = sum(pe.estimate(cr.timesteps).total_time for cr in results) / 1000

        noisy, _ = build_noisy(circuit, at_gates, anchor_ops, gaps, params)
        det, obs = noisy.compile_detector_sampler(
            seed=42_000 + (1 if carry else 0)).sample(
            SHOTS, separate_observables=True)
        pred = matcher_from(noisy).decode_batch(det)
        ler = float(np.any(pred != obs, axis=1).mean())

        all_tr = [max(p[1] for p in g["phonons"].values()) for g in at_gates]
        summary_rows.append(dict(
            config=label, mode=mode, rounds=ROUNDS, shots=SHOTS,
            ler=f"{ler:.5f}", cools=cools_total, time_ms=f"{t_ms:.1f}",
            mean_nbar=f"{statistics.mean(all_tr):.3f}",
            max_nbar=f"{max(all_tr):.3f}", gates_clipped=clipped))
        print(f"[{label:>10} | {mode:>5}] LER {ler:.5f}  cools {cools_total:3d}  "
              f"{t_ms:6.0f} ms  mean n_bar {statistics.mean(all_tr):5.2f}  "
              f"max {max(all_tr):5.2f}  clipped {clipped}  "
              f"({time.perf_counter() - t0:.0f}s)")

for name, rows in (("", summary_rows), ("_rounds", round_rows)):
    out = HERE / f"results_carryover_d{D}{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

print(f"\nd={D}, {ROUNDS} rounds, {SHOTS} shots; lookup clips at n_bar={N_MAX:g}")
print(f"{'config':<12} {'mode':<6} {'LER':>9} {'cools':>6} {'ms':>7} "
      f"{'mean n̄':>8} {'max n̄':>7} {'clipped':>8}")
for r in summary_rows:
    print(f"{r['config']:<12} {r['mode']:<6} {r['ler']:>9} {r['cools']:>6} "
          f"{r['time_ms']:>7} {r['mean_nbar']:>8} {r['max_nbar']:>7} "
          f"{r['gates_clipped']:>8}")
print("\nper-round n_bar trajectory (mean over MS gates in the round):")
for label in dict.fromkeys(r["config"] for r in round_rows):
    for mode in ("reset", "carry"):
        tr = [r["mean_nbar"] for r in round_rows
              if r["config"] == label and r["mode"] == mode]
        print(f"  {label:<12} {mode:<6} " + " ".join(f"{v:>7}" for v in tr))
