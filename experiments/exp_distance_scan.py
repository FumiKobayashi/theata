"""Scan logical error rate p_L across code distances d=3, 5, 7.

Runs end-to-end pipeline (compile → recompile → noise inject → MWPM decode)
for each distance and produces a 2D scatter plot p_L vs d with 1-sigma error
bars. Each distance uses its forward / reverse_cnot circuit pair.

Output: results/noise_injection/distance_scan_{timestamp}.{json,png}

Usage:
    python experiments/exp_distance_scan.py --shots 50000
    python experiments/exp_distance_scan.py --distances 3 5 7 --rounds 3
"""

import sys
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from pauli_lookup import PauliErrorLookup
from stim_recompiler import recompile_multi_round
from exp_noise_injection import (
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_transport_dephasing,
    default_noise_params,
)


PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"


# (forward_module_name, reverse_module_name, default_observable_qubits)
# Observable = top row (data 0..d-1), which is a horizontal Z chain — logical Z
# for the rotated surface code convention used by these snake circuits.
DISTANCE_CIRCUITS = {
    3: ('snake_reuse1', 'snake_reuse1_reverse', [0, 1, 2]),
    5: ('d5_reuse1', 'd5_reuse1_reverse_cnot', [0, 1, 2, 3, 4]),
    7: ('d7_reuse1', 'd7_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6]),
}


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
    return result


def build_noisy_circuit(forward, reverse, rounds, observable, mode='lumped'):
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
        result = _compile_round(circuit_name, current_placement, arch,
                                initial_zone_contents=current_zone_contents)
        results.append(result)
        current_placement = result.final_placement
        current_zone_contents = result.final_zone_contents

    circuit = recompile_multi_round(results, observable_column=observable)

    all_phonon_gates = []
    for r in results:
        pd = r.phonon_data or {}
        all_phonon_gates.extend(pd.get('at_gates', []))
    # PAULI_CHANNEL_2 lookup (strict): transversal phonon → 15 Pauli probs
    pauli_rates = phonon_to_pauli_lookup_rates(
        all_phonon_gates, PAULI_LOOKUP, mode=LOOKUP_MODE)
    noise_params = default_noise_params()
    noisy = inject_noise_approach_b(circuit, pauli_rates, noise_params)
    noisy = inject_transport_dephasing(noisy, results, noise_params)

    return noisy, results


def simulate(circuit, shots, seed=None):
    """Returns (n_errors, n_shots)."""
    sampler = (circuit.compile_detector_sampler(seed=seed)
               if seed is not None else circuit.compile_detector_sampler())
    # approximate_disjoint_errors=True: PAULI_CHANNEL_2 を含む回路で必須
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det_samples, obs_actual = sampler.sample(shots, separate_observables=True)
    obs_predicted = matcher.decode_batch(det_samples)
    errors_per_shot = np.any(obs_predicted != obs_actual, axis=1)
    return int(errors_per_shot.sum()), shots


def main():
    parser = argparse.ArgumentParser(description='Distance scan')
    parser.add_argument('--distances', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--shots', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='results/noise_injection',
                        help='Output directory for plot+JSON')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {
        'rounds': args.rounds,
        'shots_per_d': args.shots,
        'seed': args.seed,
        'transport_mode': 'lumped',
        'results': [],
    }

    for d in args.distances:
        if d not in DISTANCE_CIRCUITS:
            print(f"[skip] d={d} not configured")
            continue
        fwd, rev, obs = DISTANCE_CIRCUITS[d]
        print(f"=== d={d} ({fwd} / {rev}, observable={obs}) ===")
        noisy, results = build_noisy_circuit(fwd, rev, args.rounds, obs)
        # Quick info
        n_qubits = noisy.num_qubits
        n_inst = sum(1 for _ in noisy)
        print(f"  num_qubits: {n_qubits}, instructions: {n_inst}")

        # Decode
        n_err, n_shots = simulate(noisy, args.shots, seed=args.seed + d)
        p_L = n_err / n_shots
        stderr = math.sqrt(p_L * (1 - p_L) / n_shots) if n_shots > 0 else 0
        per_round = 1 - (1 - p_L) ** (1.0 / args.rounds) if p_L > 0 else 0
        print(f"  errors: {n_err}/{n_shots}")
        print(f"  p_L (over {args.rounds} rounds): {p_L:.4e} +/- {stderr:.4e}")
        print(f"  per-round (approx):              {per_round:.4e}")

        data['results'].append({
            'd': d,
            'forward': fwd,
            'reverse': rev,
            'observable': obs,
            'errors': n_err,
            'shots': n_shots,
            'p_L': p_L,
            'stderr_1sigma': stderr,
            'per_round_p_L': per_round,
        })

    # Save JSON
    json_path = outdir / f'distance_scan_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot
    if not data['results']:
        print("[skip plot] no results")
        return 0

    ds = [r['d'] for r in data['results']]
    pLs = [r['p_L'] for r in data['results']]
    errs = [r['stderr_1sigma'] for r in data['results']]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ds, pLs, yerr=errs, fmt='o-', capsize=4, markersize=8,
                color='C0', label=f'{args.rounds} rounds, {args.shots} shots/d')
    ax.set_xlabel('Code distance d')
    ax.set_ylabel(r'Logical error rate $p_L$')
    ax.set_yscale('log')
    ax.set_xticks(ds)
    ax.set_title(f'Logical error vs distance (d={ds}, {args.rounds} rounds, '
                 f'snake_reuse1 family)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = outdir / f'distance_scan_{timestamp}.png'
    fig.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
