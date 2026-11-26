# -*- coding: utf-8 -*-
"""Theta sweep: can realization knowledge (genie) buy back weakened cooling?

For each cooling setting (no cooling / theta=3.0 / theta=1.5), generate K
realization circuits (n~_g = n_bar_g * Lognormal(1, kappa)) and decode each
with (i) the mean-DEM decoder and (ii) its own genie DEM. Reports, per theta:
mean gate-time n_bar, cooling count, estimated wall time, and LER
mean +/- spread across realizations for both decoders.

Research question: how close does genie@weak-cooling get to mean@strong-cooling
(trading cooling time for decoder knowledge), and where does the genie gap open.

Usage: venv\\Scripts\\python.exe hackathon\\eval_theta_sweep.py
Results: hackathon/results_theta_sweep.csv (per-realization LERs included)
"""
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_test_circuits import compile_rounds, scaled_at_gates, build_noisy  # noqa: E402

from stim_recompiler import recompile_multi_round  # noqa: E402
from ion_timeline import gaps_from_compile_results  # noqa: E402
from exp_noise_injection import default_noise_params  # noqa: E402
from parallel_estimator import ParallelEstimator  # noqa: E402
import pymatching  # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--d", type=int, default=3)
_ap.add_argument("--rounds", type=int, default=3)
_ap.add_argument("--kappa", type=float, default=0.3)
_ap.add_argument("--K", type=int, default=50)
_ap.add_argument("--shots", type=int, default=20_000)
_args = _ap.parse_args()
D, ROUNDS = _args.d, _args.rounds
KAPPA, K, SHOTS = _args.kappa, _args.K, _args.shots
THETAS = [None, 3.0, 1.5]


def matcher_from(c):
    dem = c.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


sigma = math.sqrt(math.log(1.0 + KAPPA ** 2))
mu = -0.5 * sigma ** 2

out_rows = []
summary = []
for ti, theta in enumerate(THETAS):
    label = "no cooling" if theta is None else f"theta={theta:g}"
    t0 = time.perf_counter()
    results, observable = compile_rounds(D, ROUNDS, theta)
    circuit = recompile_multi_round(results, observable_column=observable)
    anchor_ops, gaps = gaps_from_compile_results(results)
    params = default_noise_params()
    at_gates = []
    for cr in results:
        at_gates.extend((cr.phonon_data or {}).get("at_gates", []))
    n_gates = len(at_gates)
    mean_tr = [max(p[1] for p in g["phonons"].values()) for g in at_gates]
    cools = sum(cr.stats.get("cools", 0) for cr in results)
    pe = ParallelEstimator()
    t_ms = sum(pe.estimate(cr.timesteps).total_time for cr in results) / 1000

    mean_noisy, _ = build_noisy(circuit, at_gates, anchor_ops, gaps, params)
    m_mean = matcher_from(mean_noisy)

    rng = np.random.default_rng(1000 + ti)
    lers_mean, lers_genie = [], []
    for k in range(K):
        factors = rng.lognormal(mu, sigma, size=n_gates)
        noisy_k, _ = build_noisy(
            circuit, scaled_at_gates(at_gates, factors),
            anchor_ops, gaps, params)
        det, obs = noisy_k.compile_detector_sampler(seed=5000 + 100 * ti + k).sample(
            SHOTS, separate_observables=True)
        pred = m_mean.decode_batch(det)
        lm = float(np.any(pred != obs, axis=1).mean())
        pred = matcher_from(noisy_k).decode_batch(det)
        lg = float(np.any(pred != obs, axis=1).mean())
        lers_mean.append(lm)
        lers_genie.append(lg)
        out_rows.append((label, k, f"{lm:.5f}", f"{lg:.5f}"))

    summary.append((label, statistics.mean(mean_tr), cools, t_ms,
                    statistics.mean(lers_mean), statistics.pstdev(lers_mean),
                    statistics.mean(lers_genie), statistics.pstdev(lers_genie)))
    print(f"[{label}] done in {time.perf_counter() - t0:.0f}s  "
          f"(avg gate n_bar {statistics.mean(mean_tr):.2f}, cools {cools}, "
          f"{t_ms:.0f} ms/3 rounds)")

with open(HERE / f"results_theta_sweep_d{D}_k{KAPPA:g}.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["config", "realization", "ler_mean_dem", "ler_genie_dem"])
    w.writerows(out_rows)

print()
print(f"d={D}, {ROUNDS} rounds, kappa={KAPPA}, K={K}, {SHOTS} shots/realization")
print(f"{'config':<12} {'avg n̄':>7} {'cools':>5} {'ms/3r':>7} "
      f"{'LER mean-DEM':>16} {'LER genie':>16} {'gap':>7}")
for label, nbar, cools, t_ms, lm, sm, lg, sg in summary:
    gap = (lm - lg) / lm * 100 if lm > 0 else 0.0
    print(f"{label:<12} {nbar:>7.2f} {cools:>5d} {t_ms:>7.0f} "
          f"{lm:>9.4f} ±{sm:.4f} {lg:>9.4f} ±{sg:.4f} {gap:>6.1f}%")
print("(± = spread across realizations, i.e. the LER variance of the TODO)")
