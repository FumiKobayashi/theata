# -*- coding: utf-8 -*-
"""Test-circuit ensemble generator for the soft-decoder hackathon project.

Generates K Stim circuits that share one compiled schedule but differ in
the PAULI_CHANNEL_2 probabilities: each circuit is one *realization* of
per-gate phonon numbers  n~_g = n_bar_g * X_g,  X_g ~ Lognormal(mean 1, CV kappa).
The tracked n_bar is thereby demoted from "the value" to "the mean of an
assumed distribution" (see NBAR_VARIANCE_NOTE.md, definition (b)).

Outputs, under test_circuits/d{d}_r{rounds}_k{kappa}_s{seed}/:
  mean.stim          the mean-n_bar circuit (source of the baseline DEM)
  r000.stim ...      K realization circuits (the test data)
  realizations.csv   per gate and per realization: mean n_bar, factor X,
                     realized n_bar, lookup key actually used (strict/ceil)
Channel evaluation uses the existing lookup in strict mode (0.25 grid,
fine for kappa >= 0.3; implement interpolation before trusting kappa = 0.1).

Usage:
  venv\\Scripts\\python.exe hackathon\\make_test_circuits.py
  venv\\Scripts\\python.exe hackathon\\make_test_circuits.py --d 3 --kappa 0.3 --K 20 --seed 42
  (optional: --cool-threshold 1.5 to generate under the simple threshold policy)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent
for sub in ("src", "circuits", "experiments"):
    sys.path.insert(0, str(REPO / sub))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stim_recompiler import recompile_multi_round
from ion_timeline import gaps_from_compile_results
from pauli_lookup import PauliErrorLookup
from exp_noise_injection import (
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_per_ion_idle,
    default_noise_params,
)
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt

LOOKUP = PauliErrorLookup()


def compile_rounds(d, rounds, cool_threshold=None, carry_phonons=False):
    """carry_phonons=True で前ラウンドの最終 per-ion フォノンを次ラウンドに
    引き継ぐ (既定 False = 論文設定のラウンド境界リセット)。"""
    forward, reverse, observable = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    placement = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)
    kwargs = {}
    if cool_threshold is not None:
        kwargs["cool_threshold_transversal"] = cool_threshold
    cfg = CompilerConfig(**kwargs)
    results, zone_contents, prev_phonons = [], None, None
    for r in range(1, rounds + 1):
        name = forward if r % 2 == 1 else reverse
        m = __import__(name)
        dag = DAGBuilder(parse(m.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        ckw = {}
        if carry_phonons and prev_phonons is not None:
            ckw["initial_phonons"] = prev_phonons
        cr = QCCDCompiler(arch, config=cfg).compile(dag, placement, **ckw)
        results.append(cr)
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
        pd = cr.phonon_data or {}
        prev_phonons = {"axial": pd.get("per_ion_axial", {}),
                        "transversal": pd.get("per_ion_transversal", {}),
                        "transversal2": pd.get("per_ion_transversal2", {})}
    return results, observable


def scaled_at_gates(at_gates, factors):
    """Copy of at_gates with each gate's transversal phonons scaled by its factor.

    The lookup consumes the transversal component (index 1); the axial
    component is kept untouched (recorded only).
    """
    out = []
    for g, x in zip(at_gates, factors):
        ph = {ion: tuple(v * x if i == 1 else v for i, v in enumerate(p))
              for ion, p in g["phonons"].items()}
        ng = dict(g)
        ng["phonons"] = ph
        out.append(ng)
    return out


def build_noisy(circuit, at_gates, anchor_ops, gaps, params):
    rates = phonon_to_pauli_lookup_rates(
        at_gates, LOOKUP, mode="strict", warn_on_clip=False)
    noisy = inject_noise_approach_b(circuit, rates, params, include_idle=False)
    noisy = inject_per_ion_idle(noisy, anchor_ops, gaps, params)
    return noisy, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3, choices=sorted(DISTANCE_CIRCUITS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kappa", type=float, default=0.3,
                    help="coefficient of variation of the lognormal factor X_g")
    ap.add_argument("--K", type=int, default=20, help="number of realizations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cool-threshold", type=float, default=None,
                    help="simple threshold policy theta (default: no cooling)")
    args = ap.parse_args()

    tag = f"d{args.d}_r{args.rounds}_k{args.kappa:g}_s{args.seed}"
    if args.cool_threshold is not None:
        tag += f"_cool{args.cool_threshold:g}"
    out_dir = HERE / "test_circuits" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"compiling d={args.d}, {args.rounds} rounds "
          f"(cool_threshold={args.cool_threshold}) ...")
    results, observable = compile_rounds(args.d, args.rounds, args.cool_threshold)
    circuit = recompile_multi_round(results, observable_column=observable)
    anchor_ops, gaps = gaps_from_compile_results(results)
    params = default_noise_params()

    at_gates = []
    for cr in results:
        at_gates.extend((cr.phonon_data or {}).get("at_gates", []))
    n_gates = len(at_gates)
    mean_tr = [max(p[1] for p in g["phonons"].values()) for g in at_gates]

    # mean circuit (baseline DEM source) = factors all 1
    mean_noisy, _ = build_noisy(circuit, at_gates, anchor_ops, gaps, params)
    (out_dir / "mean.stim").write_text(str(mean_noisy), encoding="utf-8")

    # lognormal with E[X] = 1 and CV = kappa
    sigma = math.sqrt(math.log(1.0 + args.kappa ** 2))
    mu = -0.5 * sigma ** 2
    rng = np.random.default_rng(args.seed)

    rows = []
    for k in range(args.K):
        factors = rng.lognormal(mu, sigma, size=n_gates)
        noisy, rates = build_noisy(
            circuit, scaled_at_gates(at_gates, factors),
            anchor_ops, gaps, params)
        (out_dir / f"r{k:03d}.stim").write_text(str(noisy), encoding="utf-8")
        for gi, ((gate_name, _probs, n_used, _ax, _tr), x) in enumerate(
                zip(rates, factors)):
            rows.append((k, gi, gate_name, f"{mean_tr[gi]:.4f}", f"{x:.4f}",
                         f"{mean_tr[gi] * x:.4f}", n_used))

    with open(out_dir / "realizations.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["realization", "gate_index", "gate", "n_bar_mean",
                    "factor_X", "n_bar_realized", "lookup_key_used"])
        w.writerows(rows)

    print(f"wrote {args.K} realization circuits + mean.stim + "
          f"realizations.csv -> {out_dir}")

    # --- smoke test: decode r000 with the mean DEM and with its own DEM ---
    import pymatching
    import stim

    def matcher_from(c):
        dem = c.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)
        return pymatching.Matching.from_detector_error_model(dem)

    r0 = stim.Circuit((out_dir / "r000.stim").read_text(encoding="utf-8"))
    det, obs = r0.compile_detector_sampler(seed=7).sample(
        20_000, separate_observables=True)
    for label, matcher in (("mean DEM   ", matcher_from(mean_noisy)),
                           ("genie DEM  ", matcher_from(r0))):
        pred = matcher.decode_batch(det)
        ler = float(np.any(pred != obs, axis=1).mean())
        print(f"smoke r000, {label}: LER = {ler:.4f}  (20k shots)")


if __name__ == "__main__":
    main()
