# -*- coding: utf-8 -*-
"""Shape-mismatch experiment: does the decoder need to know P(n)'s shape?

Truth circuits share the per-gate mean trajectory n_bar_g but differ in the
phonon-number distribution shape (Poisson / NB k=4 / thermal / NB k=0.5),
each mapped to exact channels through the Fock-resolved table. Every truth
is decoded with
  (i)   the thermal-assumption DEM (shape-mischaracterized decoder,
        exact n_bar -- today's pipeline upgraded to no grid),
  (ii)  its own DEM (shape-matched decoder),
  (iii) the legacy CSV lookup DEM (strict/ceil 0.25 grid, thermal) --
        quantifies the old quantization bias on the side.
Deterministic: no realization sampling, one circuit per cell, paired shots.

Usage: venv\\Scripts\\python.exe hackathon\\eval_shape_mismatch.py --d 3
Results: hackathon/results_shape_mismatch_d{d}.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_test_circuits import compile_rounds, LOOKUP  # noqa: E402
from fock_adapter import FockChannelLookup, fock_rates  # noqa: E402

from stim_recompiler import recompile_multi_round  # noqa: E402
from ion_timeline import gaps_from_compile_results  # noqa: E402
from exp_noise_injection import (  # noqa: E402
    phonon_to_pauli_lookup_rates, inject_noise_approach_b,
    inject_per_ion_idle, default_noise_params)
import pymatching  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--shots", type=int, default=200_000)
args = ap.parse_args()
D, ROUNDS, SHOTS = args.d, args.rounds, args.shots
THETAS = [None, 1.5]
SHAPES = [("poisson", "poisson"), ("nb_k4", ("nb", 4.0)),
          ("thermal", "thermal"), ("nb_k0.5", ("nb", 0.5))]


def matcher_from(c):
    dem = c.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


rows = []
for theta in THETAS:
    label = "no cooling" if theta is None else f"theta={theta:g}"
    t0 = time.perf_counter()
    results, observable = compile_rounds(D, ROUNDS, theta)
    circuit = recompile_multi_round(results, observable_column=observable)
    anchor_ops, gaps = gaps_from_compile_results(results)
    params = default_noise_params()
    at_gates = []
    for cr in results:
        at_gates.extend((cr.phonon_data or {}).get("at_gates", []))

    def build(rates):
        noisy = inject_noise_approach_b(circuit, rates, params, include_idle=False)
        return inject_per_ion_idle(noisy, anchor_ops, gaps, params)

    # decoder DEM sources
    thermal_exact = build(fock_rates(at_gates, FockChannelLookup("thermal")))
    legacy = build(phonon_to_pauli_lookup_rates(
        at_gates, LOOKUP, mode="strict", warn_on_clip=False))
    m_thermal = matcher_from(thermal_exact)
    m_legacy = matcher_from(legacy)

    for name, shape in SHAPES:
        truth = thermal_exact if shape == "thermal" else build(
            fock_rates(at_gates, FockChannelLookup(shape)))
        det, obs = truth.compile_detector_sampler(
            seed=90_000 + hash((label, name)) % 10_000).sample(
            SHOTS, separate_observables=True)
        lers = {}
        for dec, m in (("thermal_exact", m_thermal),
                       ("matched", matcher_from(truth)),
                       ("legacy_lookup", m_legacy)):
            pred = m.decode_batch(det)
            lers[dec] = float(np.any(pred != obs, axis=1).mean())
        rows.append(dict(config=label, truth_shape=name, shots=SHOTS, **lers))
        print(f"[{label:>10}] truth={name:<8} "
              f"thermal-DEM {lers['thermal_exact']:.5f}  "
              f"matched {lers['matched']:.5f}  "
              f"legacy {lers['legacy_lookup']:.5f}")
    print(f"[{label}] done in {time.perf_counter() - t0:.0f}s")

out = HERE / f"results_shape_mismatch_d{D}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}")

print(f"\nd={D}, {ROUNDS} rounds, {SHOTS} shots (paired across decoders)")
print(f"{'config':<12} {'truth':<8} {'LER thermal-DEM':>16} {'LER matched':>12} "
      f"{'mismatch gap':>13} {'LER legacy':>11}")
for r in rows:
    gap = ((r["thermal_exact"] - r["matched"]) / r["thermal_exact"] * 100
           if r["thermal_exact"] > 0 else 0.0)
    print(f"{r['config']:<12} {r['truth_shape']:<8} {r['thermal_exact']:>16.5f} "
          f"{r['matched']:>12.5f} {gap:>12.1f}% {r['legacy_lookup']:>11.5f}")
