"""Stim simulation of the noisy recompiled circuit, with MWPM decoding.

End-to-end pipeline:
  1. Compile multi-round QCCD circuit (forward / reverse alternating)
  2. Recompile to logical Stim with DETECTOR + OBSERVABLE_INCLUDE
  3. Inject noise (Approach B) and transport-time dephasing
  4. Build Detector Error Model (DEM)
  5. Decode samples with pymatching (MWPM)
  6. Report per-shot logical error rate (decoded vs ground-truth observable)

Usage:
    python experiments/exp_stim_simulation.py --rounds 3 --shots 10000
    python experiments/exp_stim_simulation.py --transport-mode per-op --shots 100000

Outputs metrics for the logical Z observable (default: top row of d=3 surface
code = data qubits 0, 1, 2).
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import math

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np
import stim
import pymatching

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stim_recompiler import recompile_multi_round
from exp_noise_injection import (
    phonon_to_error_rates,
    inject_noise_approach_b,
    inject_transport_dephasing,
    inject_per_op_transport_dephasing,
    _per_round_dephasing_time_us,
    default_noise_params,
)


def _compile_round(circuit_name, initial_placement, arch, initial_zone_contents=None):
    mod = __import__(circuit_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    if initial_zone_contents is not None:
        ordered = {}
        for zone_id in sorted(initial_zone_contents.keys()):
            for ion_id in initial_zone_contents[zone_id]:
                ordered[ion_id] = zone_id
        initial_placement = ordered
    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial_placement)
    return mod, result


def build_noisy_circuit(forward, reverse, rounds, observable, transport_mode):
    """Returns (noisy_circuit, results, transport_us_per_round)."""
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)

    initial = {}
    for i, q in enumerate(fwd_mod.DATA_QUBITS):
        initial[q] = 1 if i < len(fwd_mod.DATA_QUBITS) // 2 else 5
    for a in fwd_mod.ANCILLA_QUBITS:
        initial[a] = 5

    results = []
    current_placement = initial
    current_zone_contents = None
    for r in range(1, rounds + 1):
        circuit_name = forward if r % 2 == 1 else reverse
        _, result = _compile_round(circuit_name, current_placement, arch,
                                   initial_zone_contents=current_zone_contents)
        results.append(result)
        current_placement = result.final_placement
        current_zone_contents = result.final_zone_contents

    # Recompile to Stim with DETECTOR / OBSERVABLE_INCLUDE
    circuit = recompile_multi_round(results, observable_column=observable)

    # Inject noise
    all_phonon_gates = []
    for r in results:
        pd = r.phonon_data or {}
        all_phonon_gates.extend(pd.get('at_gates', []))
    error_rates = phonon_to_error_rates(all_phonon_gates)
    noise_params = default_noise_params()
    noisy = inject_noise_approach_b(circuit, error_rates, noise_params)

    # Transport-time dephasing
    transport_us = [_per_round_dephasing_time_us(cr) for cr in results]
    if transport_mode == 'lumped':
        noisy = inject_transport_dephasing(noisy, results, noise_params)
    elif transport_mode == 'per-op':
        noisy = inject_per_op_transport_dephasing(noisy, results, noise_params)
    # else: 'none' → no transport noise

    return noisy, results, transport_us


def decode_and_count_errors(circuit, shots, seed=None):
    """Sample, decode with MWPM, return (errors, shots) tuple."""
    if seed is not None:
        sampler = circuit.compile_detector_sampler(seed=seed)
    else:
        sampler = circuit.compile_detector_sampler()

    # Detector error model for MWPM
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    det_samples, obs_actual = sampler.sample(shots, separate_observables=True)
    obs_predicted = matcher.decode_batch(det_samples)

    # XOR predicted vs actual; any nonzero = logical error
    errors_per_shot = np.any(obs_predicted != obs_actual, axis=1)
    n_errors = int(errors_per_shot.sum())
    return n_errors, shots


def main():
    parser = argparse.ArgumentParser(description='Stim simulation + MWPM decode')
    parser.add_argument('--forward', default='snake_reuse1')
    parser.add_argument('--reverse', default='snake_reuse1_reverse')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--observable', nargs='*', type=int, default=[0, 1, 2],
                        help='Data qubit IDs for logical Z observable')
    parser.add_argument('--transport-mode', choices=['lumped', 'per-op', 'none'],
                        default='lumped')
    parser.add_argument('--shots', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dump', metavar='PATH',
                        help='Save the noisy Stim circuit to file')
    parser.add_argument('--compare-modes', action='store_true',
                        help='Run both lumped and per-op modes for direct comparison')
    args = parser.parse_args()

    modes = ['lumped', 'per-op'] if args.compare_modes else [args.transport_mode]

    print(f"Circuit: forward={args.forward}, reverse={args.reverse}, rounds={args.rounds}")
    print(f"Observable: {args.observable}")
    print(f"Shots per mode: {args.shots}")
    print()

    for mode in modes:
        print(f"=== transport-mode: {mode} ===")
        noisy, results, transport_us = build_noisy_circuit(
            args.forward, args.reverse, args.rounds, args.observable, mode)

        # Quick stats
        n_inst = sum(1 for _ in noisy)
        n_det = sum(1 for inst in noisy if inst.name == 'DETECTOR')
        n_obs = sum(1 for inst in noisy if inst.name == 'OBSERVABLE_INCLUDE')
        print(f"  Noisy circuit: instructions={n_inst}, DETECTOR={n_det}, OBSERVABLE_INCLUDE={n_obs}")
        print(f"  Wait per round (us): {[f'{t:.1f}' for t in transport_us]}")

        # Decode
        n_errors, n_shots = decode_and_count_errors(noisy, args.shots, seed=args.seed)
        p_L = n_errors / n_shots
        # Wilson-ish stderr
        stderr = math.sqrt(p_L * (1 - p_L) / n_shots) if n_shots > 0 else 0
        print(f"  MWPM decoding:")
        print(f"    Shots: {n_shots}, logical errors: {n_errors}")
        print(f"    Logical error rate p_L = {p_L:.4e}  +/- {stderr:.4e}  (1-sigma)")
        # Per-round logical error rate (approx)
        if args.rounds > 0 and p_L > 0:
            per_round = 1 - (1 - p_L) ** (1.0 / args.rounds)
            print(f"    Per-round (approx): {per_round:.4e}")
        print()

        if args.dump and mode == args.transport_mode:
            out = Path(args.dump)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w', encoding='utf-8') as f:
                f.write(str(noisy))
            print(f"  Saved noisy circuit: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
