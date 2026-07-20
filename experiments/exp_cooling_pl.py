"""閾値ベース自動冷却の p_L 実験 (1-LIZ)。

ポリシー: MS 実行直前にゲート対象チェーンの transversal フォノンが閾値を
超えていたら LIZ に CoolOp を挿入 (cool_threshold_transversal)。
「ゲート時フォノンの最大値を閾値以下に抑える」の実装。

トレードオフの構造:
  + MS の PAULI_CHANNEL_2 エラー率が下がる (lookup 入力 n_bar が閾値以下に)
  - 冷却 1 回 = cool (デフォルト 400μs) がラウンド時間に上乗せされ、
    全 qubit の idle dephasing (per-ion timeline 経由) が増える

Usage:
    python experiments/exp_cooling_pl.py --threshold 5.0 --rounds 3 --shots 50000
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

import numpy as np
import stim
import pymatching

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stim_recompiler import recompile_multi_round
from parallel_estimator import ParallelEstimator
from ion_timeline import gaps_from_compile_results
from sim_params import load_time_parameters
from pauli_lookup import PauliErrorLookup
from exp_noise_injection import (
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_per_ion_idle,
    default_noise_params,
)
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt

PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"
D5_OBSERVABLE = [0, 1, 2, 3, 4]


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compile_rounds(cfg_kwargs, rounds):
    forward, reverse, _ = DISTANCE_CIRCUITS[5]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    cfg = CompilerConfig(**cfg_kwargs)
    results = []
    placement, zone_contents = initial, None
    for r in range(1, rounds + 1):
        name = forward if r % 2 == 1 else reverse
        mod = __import__(name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=cfg).compile(dag, placement)
        results.append(cr)
        placement = cr.final_placement
        zone_contents = cr.final_zone_contents
    return results


def run_arm(label, cfg_kwargs, rounds, shots, seed, params, tp=None):
    results = compile_rounds(cfg_kwargs, rounds)

    n_cools = sum(cr.stats.get('cools', 0) for cr in results)
    pe = ParallelEstimator(tp) if tp else ParallelEstimator()
    round_ms = [pe.estimate(cr.timesteps).total_time / 1000 for cr in results]

    all_gates = []
    for cr in results:
        all_gates.extend((cr.phonon_data or {}).get('at_gates', []))
    max_tr_at_gate = max(max(p[1] for p in g['phonons'].values())
                         for g in all_gates)

    circuit = recompile_multi_round(results, observable_column=D5_OBSERVABLE)
    det, obs = circuit.compile_detector_sampler().sample(
        200, separate_observables=True)
    assert det.sum() == 0 and obs.sum() == 0, "noiseless detectors non-zero"

    anchor_ops, gaps = gaps_from_compile_results(results, time_params=tp)
    pauli_rates = phonon_to_pauli_lookup_rates(
        all_gates, PAULI_LOOKUP, mode=LOOKUP_MODE)
    noisy = inject_noise_approach_b(circuit, pauli_rates, params,
                                    include_idle=False)
    noisy = inject_per_ion_idle(noisy, anchor_ops, gaps, params)

    dem = noisy.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = noisy.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    pred = matching.decode_batch(det)
    n_err = int(np.sum(np.any(pred != obs, axis=1)))

    print(f"  {label:10s}: cools={n_cools:3d}  "
          f"round_ms={['%.1f' % m for m in round_ms]}  "
          f"max_tr@gate={max_tr_at_gate:6.2f}  "
          f"p_L={n_err/shots:.6f} ({n_err}/{shots})")
    return {
        'label': label, 'config': cfg_kwargs, 'n_cools': n_cools,
        'round_ms': round_ms, 'wall_ms': sum(round_ms),
        'max_tr_at_gate': max_tr_at_gate,
        'n_err': n_err, 'shots': shots, 'p_l': n_err / shots,
        'ci': list(wilson_interval(n_err, shots)),
    }


def main():
    ap = argparse.ArgumentParser(description='Cooling policy p_L comparison (1-LIZ, d=5)')
    ap.add_argument('--threshold', type=float, default=None,
                    help='MS 直前閾値冷却の transversal 閾値 (例: 5.0)。省略で arm なし')
    ap.add_argument('--ancilla-cool-us', type=float, default=None,
                    help='アンシラ測定後冷却の duration μs (例: 2000)。省略で arm なし')
    ap.add_argument('--cool-us', type=float, default=None,
                    help='閾値冷却 1 回の時間 μs (TimeParameters.cool を上書き、'
                         'デフォルト 400)')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--shots', type=int, default=50_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out-dir', default='results/per_ion_noise')
    args = ap.parse_args()

    params = default_noise_params()
    tp = (load_time_parameters(overrides={'cool': args.cool_us})
          if args.cool_us is not None else None)
    cool_us = args.cool_us if args.cool_us is not None else 400.0
    print(f"1-LIZ d=5, rounds={args.rounds}, shots={args.shots}, "
          f"threshold-cool duration={cool_us:g}us")

    arms = [("no-cool", {})]
    if args.ancilla_cool_us is not None:
        arms.append((f"anc@{args.ancilla_cool_us:g}us",
                     {'cool_ancilla_after_measure_us': args.ancilla_cool_us}))
    if args.threshold is not None:
        arms.append((f"thr@{args.threshold:g}/{cool_us:g}us",
                     {'cool_threshold_transversal': args.threshold}))

    rows = [run_arm(label, kw, args.rounds, args.shots, args.seed, params, tp=tp)
            for label, kw in arms]

    print(f"\n{'='*72}")
    print(f"  {'arm':10s} {'p_L':>10s} {'95% CI':>24s} {'wall(ms)':>9s} "
          f"{'cools':>6s} {'max_tr@gate':>12s}")
    for row in rows:
        lo, hi = row['ci']
        print(f"  {row['label']:10s} {row['p_l']:10.6f} "
              f"[{lo:10.6f}, {hi:10.6f}] {row['wall_ms']:9.2f} "
              f"{row['n_cools']:6d} {row['max_tr_at_gate']:12.2f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = "_".join(label for label, _ in arms[1:]) or "baseline"
    tag = tag.replace('@', '').replace('.', 'p').replace('/', '-')
    path = out / f"pl_cooling_d5_{tag}_r{args.rounds}_{stamp}.json"
    path.write_text(json.dumps({
        'rounds': args.rounds, 'shots': args.shots, 'seed': args.seed,
        'noise_params': params, 'results': rows,
    }, indent=2), encoding='utf-8')
    print(f"\n  saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
