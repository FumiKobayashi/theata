"""Multi-LIZ 論理誤り率 p_L 計測 (STORYLINE G1)。

per-LIZ stream コンパイル (複数ラウンド、牛耕逆走反転) を Stim へ接続し、
per-ion timeline noise injection + MWPM decoding で p_L を測る。
比較用に 1-LIZ (distance-scan canonical: d5_reuse1 系) も同一ノイズモデル・
同一 per-ion 方式で計測する。

パイプライン (multi-LIZ arm):
  1. round r: compile_per_liz(scan_reverse=(r 偶数),
              initial_zone_contents=前ラウンド終状態) → estimate_streams
              → validate_merged (各ラウンド PASS 必須)
  2. merged_flat (canonical linearization) を recompile_multi_round へ
     → DETECTOR / OBSERVABLE 付き Stim 回路
  3. noiseless detector 検証 (全ゼロ必須 — fold-up と detector 配線の正しさ)
  4. ノイズ: PauliErrorLookup による MS PAULI_CHANNEL_2 (フォノン依存)
     + per-ion timeline idle Z_ERROR (multi-LIZ で唯一正しい会計)
  5. DEM (approximate_disjoint_errors) → pymatching → p_L

Usage:
    python experiments/exp_multi_liz_pl.py --rounds 3 --shots 50000
"""

import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from types import SimpleNamespace

import numpy as np
import stim
import pymatching

from qccd_parser import parse
from qccd_compiler import QCCDArchitecture, CompilerConfig
from per_liz_compiler import compile_per_liz, estimate_streams, validate_merged
from stim_recompiler import recompile_multi_round
from ion_timeline import gaps_from_per_liz_results, gaps_from_compile_results
from pauli_lookup import PauliErrorLookup
from exp_noise_injection import (
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_per_ion_idle,
    default_noise_params,
)

PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"

# d=5 rotated surface code: 論理 Z = 上段行の Z チェーン (distance scan と同一規約)
D5_OBSERVABLE = [0, 1, 2, 3, 4]


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def check_noiseless(circuit: stim.Circuit, shots: int = 200) -> bool:
    det, obs = circuit.compile_detector_sampler().sample(
        shots, separate_observables=True)
    return det.sum() == 0 and obs.sum() == 0


def decode_p_l(circuit: stim.Circuit, shots: int, seed: int):
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    pred = matching.decode_batch(det)
    n_err = int(np.sum(np.any(pred != obs, axis=1)))
    return n_err


def inject_noise_per_ion(circuit, all_phonon_gates, anchor_ops, gaps, params):
    """canonical ノイズモデル (PAULI_CHANNEL_2) + per-ion idle。"""
    pauli_rates = phonon_to_pauli_lookup_rates(
        all_phonon_gates, PAULI_LOOKUP, mode=LOOKUP_MODE)
    noisy = inject_noise_approach_b(circuit, pauli_rates, params,
                                    include_idle=False)
    return inject_per_ion_idle(noisy, anchor_ops, gaps, params)


# ==========================================
# Multi-LIZ arm
# ==========================================

def run_multi_liz(circuit_name, num_liz, rounds, shots, seed, params):
    mod = __import__(circuit_name)
    program = parse(mod.CIRCUIT)
    arch = QCCDArchitecture.chained_mlm(
        num_liz=num_liz, liz_capacity=4, memory_capacity=20)

    results = []
    zone_contents = None
    print(f"[multi-LIZ {num_liz}] compiling {rounds} rounds "
          f"({circuit_name}, boustrophedon + 3tier_usage, 反転連結)...")
    for r in range(1, rounds + 1):
        res = compile_per_liz(
            program=program, ancilla_qubits=mod.ANCILLA_QUBITS,
            data_qubits=mod.DATA_QUBITS, arch=arch, code_distance=5,
            config=CompilerConfig(eviction_strategy="balanced"),
            stab_order='boustrophedon', placement_mode='3tier_usage',
            scan_reverse=(r % 2 == 0),
            initial_zone_contents=zone_contents)
        estimate_streams(res)
        v = validate_merged(res)
        assert v.is_valid, f"round {r} validation FAILED:\n{v.summary()}"
        n_transport = sum(
            1 for op in res.merged_flat
            if type(op).__name__ in ('SplitOp', 'MergeOp', 'ShuttleOp', 'SwapOp'))
        print(f"  round {r} ({'rev' if r % 2 == 0 else 'fwd'}): "
              f"wall={res.wall_clock_us/1000:7.2f} ms  "
              f"transport_ops={n_transport}  validation=PASS")
        results.append(res)
        zone_contents = {z: list(ions)
                         for z, ions in res.state.zone_contents.items() if ions}

    # merged_flat → Stim (recompiler は flat_ops しか見ない)
    shims = [SimpleNamespace(flat_ops=r.merged_flat) for r in results]
    circuit = recompile_multi_round(shims, observable_column=D5_OBSERVABLE)
    print(f"  Stim: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors")

    ok = check_noiseless(circuit)
    print(f"  noiseless detector check: {'PASS (all zero)' if ok else 'FAIL'}")
    assert ok, "noiseless detectors non-zero — fold-up / detector wiring bug"

    anchor_ops, gaps = gaps_from_per_liz_results(results)
    all_gates = []
    for r in results:
        all_gates.extend(r.state.phonon_at_gates)
    noisy = inject_noise_per_ion(circuit, all_gates, anchor_ops, gaps, params)

    n_err = decode_p_l(noisy, shots, seed)
    wall_ms = sum(r.wall_clock_us for r in results) / 1000
    ph_ax = max(max(r.state.phonon_axial.values()) for r in results)
    return {
        'arm': f'{num_liz}-LIZ', 'circuit': circuit_name,
        'n_err': n_err, 'shots': shots, 'p_l': n_err / shots,
        'wall_ms': wall_ms, 'max_axial_phonon': ph_ax,
        'per_round_wall_ms': [r.wall_clock_us / 1000 for r in results],
    }


# ==========================================
# 1-LIZ arm (distance-scan canonical + per-ion noise)
# ==========================================

def run_single_liz(rounds, shots, seed, params):
    from exp_distance_scan import DISTANCE_CIRCUITS, _compile_round
    from exp_multi_round import placement_greedy_alt

    forward, reverse, _obs = DISTANCE_CIRCUITS[5]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    print(f"[1-LIZ] compiling {rounds} rounds ({forward} / {reverse})...")
    results = []
    placement, zone_contents = initial, None
    for r in range(1, rounds + 1):
        name = forward if r % 2 == 1 else reverse
        cr = _compile_round(name, placement, arch,
                            initial_zone_contents=zone_contents)
        results.append(cr)
        placement = cr.final_placement
        zone_contents = cr.final_zone_contents

    circuit = recompile_multi_round(results, observable_column=D5_OBSERVABLE)
    ok = check_noiseless(circuit)
    print(f"  noiseless detector check: {'PASS (all zero)' if ok else 'FAIL'}")
    assert ok

    anchor_ops, gaps = gaps_from_compile_results(results)
    all_gates = []
    for cr in results:
        all_gates.extend((cr.phonon_data or {}).get('at_gates', []))
    noisy = inject_noise_per_ion(circuit, all_gates, anchor_ops, gaps, params)

    n_err = decode_p_l(noisy, shots, seed)
    from parallel_estimator import ParallelEstimator
    pe = ParallelEstimator()
    wall_ms = sum(pe.estimate(cr.timesteps).total_time for cr in results) / 1000
    ph_ax = max((cr.phonon_data or {}).get('max_axial', 0.0) for cr in results)
    return {
        'arm': '1-LIZ', 'circuit': forward,
        'n_err': n_err, 'shots': shots, 'p_l': n_err / shots,
        'wall_ms': wall_ms, 'max_axial_phonon': ph_ax,
        'per_round_wall_ms': [pe.estimate(cr.timesteps).total_time / 1000
                              for cr in results],
    }


def main():
    ap = argparse.ArgumentParser(description='Multi-LIZ p_L measurement (G1)')
    ap.add_argument('--circuit', default='d5_reuse2')
    ap.add_argument('--num-liz', type=int, default=2)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--shots', type=int, default=50_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--skip-single', action='store_true')
    ap.add_argument('--out-dir', default='results/per_ion_noise')
    args = ap.parse_args()

    params = default_noise_params()
    rows = []

    rows.append(run_multi_liz(args.circuit, args.num_liz, args.rounds,
                              args.shots, args.seed, params))
    if not args.skip_single:
        rows.append(run_single_liz(args.rounds, args.shots, args.seed, params))

    print(f"\n{'='*72}")
    print(f"p_L @ d=5, rounds={args.rounds}, shots={args.shots}")
    print(f"{'='*72}")
    print(f"  {'arm':8s} {'circuit':12s} {'p_L':>10s} {'95% CI':>22s} "
          f"{'wall(ms)':>9s} {'max_ax_ph':>10s}")
    for row in rows:
        lo, hi = wilson_interval(row['n_err'], row['shots'])
        row['ci'] = [lo, hi]
        print(f"  {row['arm']:8s} {row['circuit']:12s} "
              f"{row['p_l']:10.6f} [{lo:9.6f}, {hi:9.6f}] "
              f"{row['wall_ms']:9.2f} {row['max_axial_phonon']:10.1f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = out / f"pl_d5_{args.num_liz}liz_r{args.rounds}_{stamp}.json"
    path.write_text(json.dumps({
        'rounds': args.rounds, 'shots': args.shots, 'seed': args.seed,
        'noise_params': params, 'results': rows,
    }, indent=2), encoding='utf-8')
    print(f"\n  saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
