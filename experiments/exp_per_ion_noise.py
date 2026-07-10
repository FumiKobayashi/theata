"""Per-ion timeline noise injection — 1-LIZ 等価性検証実験。

新方式 (per-ion timeline: 各イオンの op 間ギャップ Δt に 1 個の Z_ERROR) と
現行 2 層方式 (ゲート軸 idle + per-anchor transport 残差) を、同一の
multi-round 1-LIZ コンパイル結果に適用して比較する:

  - 総 dephasing 露出時間 (qubit·μs) の一致
  - 検出器発火率・observable 反転率
  - MWPM (pymatching) decoding 後の論理誤り率 p_L

per-ion 方式は 1-LIZ でも本来正しい会計なので、両者は統計誤差内で一致するはず
(小さな系統差: 旧方式は 5-step CX / 2-step H の合成分解の公称時間 ~16μs/CX を
コンパイル済みタイムラインとは独立に計上する)。一致すれば per-ion 方式を
multi-LIZ (per-LIZ stream) へ展開する根拠になる。

Usage:
    python experiments/exp_per_ion_noise.py --rounds 3 --shots 200000
"""

import sys
import math
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np
import stim
import pymatching

from qccd_compiler import QCCDArchitecture
from stim_recompiler import recompile_multi_round
from ion_timeline import gaps_from_compile_results
from exp_noise_injection import (
    phonon_to_error_rates,
    inject_noise_approach_b,
    inject_per_op_transport_dephasing,
    inject_per_ion_idle,
    default_noise_params,
    compute_p_idle,
)
from exp_recompile_multi_round import _compile_round


def exposure_us(circuit: stim.Circuit, gamma: float) -> float:
    """回路中の全 Z_ERROR を時間に逆変換した総 dephasing 露出 (qubit·μs)。"""
    total = 0.0
    for inst in circuit:
        if inst.name != 'Z_ERROR':
            continue
        p = inst.gate_args_copy()[0]
        if p <= 0 or p >= 0.5:
            continue
        t_us = -math.log(1.0 - 2.0 * p) / (2.0 * gamma) * 1e6
        n = sum(1 for t in inst.targets_copy() if t.is_qubit_target)
        total += n * t_us
    return total


def decode_p_l(circuit: stim.Circuit, shots: int, seed: int):
    """MWPM decoding で論理誤り率を推定。(p_L, n_errors, det_fire_mean)"""
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    pred = matching.decode_batch(det)
    n_err = int(np.sum(np.any(pred != obs, axis=1)))
    return n_err / shots, n_err, float(det.mean())


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser(description='Per-ion vs legacy noise: 1-LIZ equivalence')
    ap.add_argument('--forward', default='snake_reuse1')
    ap.add_argument('--reverse', default='snake_reuse1_reverse')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--observable', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--shots', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--dump-dir', default=None,
                    help='両方式の noisy .stim を保存するディレクトリ')
    args = ap.parse_args()

    # --- multi-round 1-LIZ compile (exp_recompile_multi_round と同一手順) ---
    fwd_mod = __import__(args.forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)

    initial = {}
    for i, q in enumerate(fwd_mod.DATA_QUBITS):
        initial[q] = 1 if i < len(fwd_mod.DATA_QUBITS) // 2 else 5
    for a in fwd_mod.ANCILLA_QUBITS:
        initial[a] = 5

    print(f"Compiling {args.rounds} rounds ({args.forward} / {args.reverse})...")
    results = []
    placement, zone_contents = initial, None
    for r in range(1, args.rounds + 1):
        name = args.forward if r % 2 == 1 else args.reverse
        _, cr = _compile_round(name, placement, arch,
                               initial_zone_contents=zone_contents)
        results.append(cr)
        placement = cr.final_placement
        zone_contents = cr.final_zone_contents

    circuit = recompile_multi_round(results, observable_column=args.observable)
    print(f"  Stim: {circuit.num_qubits} qubits, "
          f"{circuit.num_detectors} detectors, {len(circuit)} instructions")

    # --- 共通のゲートエラー (フォノン由来 MS レート) ---
    all_gates = []
    for cr in results:
        all_gates.extend((cr.phonon_data or {}).get('at_gates', []))
    error_rates = phonon_to_error_rates(all_gates)
    params = default_noise_params()
    gamma = params['gamma_phi_per_s']

    # --- 旧方式: ゲート軸 idle + per-anchor transport 残差 ---
    legacy = inject_noise_approach_b(circuit, error_rates, params)
    legacy = inject_per_op_transport_dephasing(legacy, results, params)

    # --- 新方式: per-ion timeline ---
    anchor_ops, gaps = gaps_from_compile_results(results)
    per_ion = inject_noise_approach_b(circuit, error_rates, params,
                                      include_idle=False)
    per_ion = inject_per_ion_idle(per_ion, anchor_ops, gaps, params)

    def _stats(c):
        counts = {}
        for inst in c:
            counts[inst.name] = counts.get(inst.name, 0) + 1
        return counts

    print(f"\n[circuit size]")
    for label, c in (('legacy', legacy), ('per-ion', per_ion)):
        counts = _stats(c)
        z = counts.get('Z_ERROR', 0)
        print(f"  {label:8s}: instructions={len(c):5d}  Z_ERROR insts={z:5d}")

    print(f"\n[total dephasing exposure] (qubit·μs, Z_ERROR を時間に逆変換)")
    e_legacy = exposure_us(legacy, gamma)
    e_perion = exposure_us(per_ion, gamma)
    print(f"  legacy : {e_legacy:12.1f}")
    print(f"  per-ion: {e_perion:12.1f}   (ratio {e_perion/e_legacy:.4f})")

    print(f"\n[MWPM p_L] shots={args.shots}, seed={args.seed}")
    rows = []
    for label, c in (('legacy', legacy), ('per-ion', per_ion)):
        p_l, n_err, det_mean = decode_p_l(c, args.shots, args.seed)
        lo, hi = wilson_interval(n_err, args.shots)
        rows.append((label, p_l, n_err, lo, hi, det_mean))
        print(f"  {label:8s}: p_L={p_l:.6f}  ({n_err}/{args.shots})  "
              f"95% CI [{lo:.6f}, {hi:.6f}]  det_fire_mean={det_mean:.5f}")

    (_, p1, _, lo1, hi1, _), (_, p2, _, lo2, hi2, _) = rows
    overlap = not (hi1 < lo2 or hi2 < lo1)
    print(f"\n  CI overlap: {'YES — equivalent within statistics' if overlap else 'NO — investigate'}")

    if args.dump_dir:
        out = Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for label, c in (('legacy', legacy), ('per_ion', per_ion)):
            p = out / f"d3_r{args.rounds}_{label}.stim"
            p.write_text(str(c), encoding='utf-8')
            print(f"  saved: {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
