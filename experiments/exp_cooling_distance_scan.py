"""E-C6: 冷却ポリシー込み distance scan (1-LIZ, d=3/5/7) と Λ 読み出し。

各冷却 arm について d ∈ {3, 5, 7} の多ラウンド p_L を per-ion ノイズ
パイプライン (LUT strict + per-ion idle dephasing) で測り、

    cap n̄ = x, avg n̄ = y で p_L(d) = z、Λ = p_L(d) / p_L(d+2) = a

の形にまとめる。判定は Λ > 1 (CI 込み) = 「閾値達成」。n̄ は説明変数
(cap = ゲート時 transversal の最大、avg = MS ゲートごとの最大値の平均)。

Usage:
    python experiments/exp_cooling_distance_scan.py \
        --arms no-cool anc merge --merge-cool-threshold 1.5 \
        --ancilla-cool-us 2000 --shots 20000 --quiet-clip
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
from exp_cooling_pl import wilson_interval

PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"
LUT_N_MAX = 20.0


def compile_rounds_d(d, cfg_kwargs, rounds, tp=None):
    forward, reverse, _ = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    cfg = CompilerConfig(**cfg_kwargs)
    # 常駐 coolant (回路外イオン) は LIZ に初期配置する
    if cfg.resident_coolant_ids:
        liz_id = arch.get_liz_zones()[0].id
        initial = dict(initial)
        for cid in cfg.resident_coolant_ids:
            initial[cid] = liz_id
    results = []
    placement, zone_contents = initial, None
    for r in range(1, rounds + 1):
        name = forward if r % 2 == 1 else reverse
        mod = __import__(name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=cfg, time_params=tp).compile(dag, placement)
        results.append(cr)
        placement = cr.final_placement
        zone_contents = cr.final_zone_contents
    return results


def gate_phonon_stats(all_gates):
    """MS ゲートごとの (対象イオン中の最大 transversal) の系列から cap / avg。"""
    per_gate = [max(p[1] for p in g['phonons'].values()) for g in all_gates]
    arr = np.array(per_gate)
    return {
        'n_ms': int(arr.size),
        'cap_tr': float(arr.max()),
        'avg_tr': float(arr.mean()),
        'median_tr': float(np.median(arr)),
        'n_clipped': int((arr > LUT_N_MAX).sum()),
    }


def per_round_rate(p_total, rounds):
    return 1 - (1 - p_total) ** (1.0 / rounds) if 0 < p_total < 1 else p_total


def run_point(label, d, cfg_kwargs, rounds, shots, seed, params, tp,
              warn_on_clip, time_model="lockstep", ideal_phonon=False,
              nbar_cap=None, nbar_cap_mode="clamp", precompiled=None):
    results = (precompiled if precompiled is not None
               else compile_rounds_d(d, cfg_kwargs, rounds, tp=tp))
    _, _, observable = DISTANCE_CIRCUITS[d]

    all_gates = []
    for cr in results:
        all_gates.extend((cr.phonon_data or {}).get('at_gates', []))

    if ideal_phonon:
        # 加熱なし下限: コンパイラの時間構造 (輸送・idle dephasing・SPAM) は
        # そのままに、全ゲートのフォノンを 0 に置換して LUT を引く。
        # p_L のうち加熱由来の寄与を分離するための対照行。
        nbar_cap = 0.0
    if nbar_cap is not None:
        # n̄ cap 補正: 実験はそのまま実行し、記録上のフォノンを補正してから
        # LUT を引く (全 3 モード一律)。
        #   clamp: min(v, cap) — 最初 0 から蓄積し cap で頭打ち
        #          (cap=0 は ideal_phonon、cap=inf はベースラインと厳密一致)
        #   const: v ≡ cap — 最初から cap に張り付いた定常パターン
        #          (10/15 西尾先生指定の第 2 アーム。clamp との対が
        #           立ち上がり過渡と定常の上下界を与える)
        if nbar_cap_mode == "const":
            fix = lambda v: nbar_cap
        else:
            fix = lambda v: min(v, nbar_cap)
        all_gates = [
            {**g, 'phonons': {ion: tuple(fix(v) for v in vs)
                              for ion, vs in g['phonons'].items()}}
            for g in all_gates]

    n_cools = sum(cr.stats.get('cools', 0) for cr in results)
    if time_model == "event":
        from event_scheduler import schedule_events
        round_ms = [schedule_events(cr.flat_ops, tp)[1] / 1000
                    for cr in results]
    else:
        pe = ParallelEstimator(tp) if tp else ParallelEstimator()
        round_ms = [pe.estimate(cr.timesteps).total_time / 1000
                    for cr in results]
    stats = gate_phonon_stats(all_gates)

    circuit = recompile_multi_round(results, observable_column=observable)
    det, obs = circuit.compile_detector_sampler().sample(
        200, separate_observables=True)
    assert det.sum() == 0 and obs.sum() == 0, "noiseless detectors non-zero"

    anchor_ops, gaps = gaps_from_compile_results(results, time_params=tp,
                                                 mode=time_model)
    pauli_rates = phonon_to_pauli_lookup_rates(
        all_gates, PAULI_LOOKUP, mode=LOOKUP_MODE, warn_on_clip=warn_on_clip)
    noisy = inject_noise_approach_b(circuit, pauli_rates, params,
                                    include_idle=False)
    noisy = inject_per_ion_idle(noisy, anchor_ops, gaps, params)

    dem = noisy.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = noisy.compile_detector_sampler(seed=seed + d).sample(
        shots, separate_observables=True)
    pred = matching.decode_batch(det)
    n_err = int(np.sum(np.any(pred != obs, axis=1)))
    p_total = n_err / shots
    lo, hi = wilson_interval(n_err, shots)

    clip_note = f"  ⚠clipped={stats['n_clipped']}" if stats['n_clipped'] else ""
    print(f"  {label:12s} d={d}: cap={stats['cap_tr']:6.2f} "
          f"avg={stats['avg_tr']:5.2f}  cools={n_cools:4d}  "
          f"wall={sum(round_ms):7.1f}ms  p_L={p_total:.5f} "
          f"[{lo:.5f},{hi:.5f}] ({n_err}/{shots}){clip_note}")
    return {
        'label': label, 'd': d, 'config': cfg_kwargs, 'rounds': rounds,
        'nbar_cap': nbar_cap,
        'n_cools': n_cools, 'round_ms': round_ms, 'wall_ms': sum(round_ms),
        **stats,
        'n_err': n_err, 'shots': shots,
        'p_L': p_total, 'ci': [lo, hi],
        'p_L_per_round': per_round_rate(p_total, rounds),
    }


def lambda_ratio(row_a, row_b):
    """Λ = p_L(d_a)/p_L(d_b) (d_b = d_a + 2)。総 p_L ベース + 1σ の対数伝播。"""
    pa, na, sa = row_a['p_L'], row_a['shots'], row_a['n_err']
    pb, nb, sb = row_b['p_L'], row_b['shots'], row_b['n_err']
    if sa == 0 or sb == 0:
        return None
    lam = pa / pb
    se_ln = math.sqrt((1 - pa) / sa + (1 - pb) / sb)  # binomial, log scale
    lam_pr = row_a['p_L_per_round'] / row_b['p_L_per_round']
    return {
        'lambda': lam,
        'ci_1sigma': [lam * math.exp(-se_ln), lam * math.exp(se_ln)],
        'ci_2sigma': [lam * math.exp(-2 * se_ln), lam * math.exp(2 * se_ln)],
        'lambda_per_round': lam_pr,
    }


def main():
    ap = argparse.ArgumentParser(
        description='E-C6: cooling-policy distance scan with Λ readout')
    ap.add_argument('--arms', nargs='+', default=['no-cool', 'anc', 'merge'],
                    choices=['no-cool', 'anc', 'merge', 'thr', 'pred',
                             'resident', 'resident-pred', 'pred-data'],
                    help='no-cool / anc (アンシラ測定後冷却) / merge (merge '
                         '平均化冷却 + anc) / thr (ゲート前閾値チェーン冷却) / '
                         'pred (予測型 merge 冷却 + anc, 2-pass) / '
                         'resident (LIZ 常駐 coolant + 補助一斉再冷却 + anc)')
    ap.add_argument('--distances', nargs='+', type=int, default=[3, 5, 7])
    ap.add_argument('--ancilla-cool-us', type=float, default=2000.0)
    ap.add_argument('--merge-cool-threshold', type=float, default=1.5)
    ap.add_argument('--threshold', type=float, default=5.0,
                    help='thr arm の cool_threshold_transversal')
    ap.add_argument('--pred-threshold', type=float, default=1.5,
                    help='pred arm の predictive_cool_threshold_transversal')
    ap.add_argument('--pred-max-iters', type=int, default=1,
                    help='pred arm の 1 チェックポイントあたりエピソード上限')
    ap.add_argument('--resident-coolants', type=int, default=2,
                    help='resident arm の LIZ 常駐 coolant 数')
    ap.add_argument('--coolant-recool-us', type=float, default=400.0,
                    help='resident arm の補助一斉再冷却時間 (μs)')
    ap.add_argument('--cool-us', type=float, default=None)
    ap.add_argument('--params', default=None,
                    help='TimeParameters の JSON (例: params/mainz_heating.json)。'
                         'heating rate 等がコンパイラのフォノン追跡に反映される')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--shots', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--quiet-clip', action='store_true')
    ap.add_argument('--out-dir', default='results/cooling_distance_scan')
    ap.add_argument('--planner', choices=['v1', 'v2'], default='v1',
                    help='v2 = per-zone state machine + JIT staging (planner v2)')
    ap.add_argument('--prefetch-depth', type=int, default=0)
    ap.add_argument('--v2-sort-horizon', type=int, default=2)
    ap.add_argument('--v2-sort-trigger', default='measure',
                    choices=['measure', 'delivery', 'both'])
    ap.add_argument('--isolate-recool', action='store_true',
                    help='窓内の合法再冷却: もつれ保持中アンシラを memory へ'
                         '一時退避して coolant のみ冷却し呼び戻す (10/25 著者案)')
    ap.add_argument('--isolate-recool-iters', type=int, default=1,
                    help='隔離サイクル (退避→Cool→帰還) の反復数 (既定 1)')
    ap.add_argument('--cool-residual', type=float, default=None,
                    help='冷却後の残留フォノン数 (axial/tr1/tr2 すべてに適用)。'
                         '抽象 Cool 命令の (t, n̄_res) 要件スイープ用')
    ap.add_argument('--ideal-phonon', action='store_true',
                    help='全ゲートのフォノンを 0 に置換して LUT を引く '
                         '(加熱なし下限。時間構造・idle dephasing は不変)')
    ap.add_argument('--nbar-cap', nargs='+', type=float, default=None,
                    metavar='CAP',
                    help='n̄ cap スイープ: 実験はそのまま実行し、記録上 cap 超過分を '
                         'cap に補正してから LUT を引く (LER vs n̄ cap 用、通常 '
                         '--arms no-cool と併用)。arm×d ごとにコンパイルを 1 回に '
                         'まとめて cap を一括評価。inf 可 (=補正なしベースライン)')
    ap.add_argument('--nbar-cap-mode', choices=['clamp', 'const'],
                    default='clamp',
                    help='clamp = min(n̄, cap) (初期 0 から蓄積・頭打ち) / '
                         'const = n̄ ≡ cap (最初から cap の定常パターン)')
    ap.add_argument('--time-model', choices=['lockstep', 'event'],
                    default='lockstep',
                    help='event = イベント駆動時間で wall と per-ion idle '
                         '(dephasing) を評価する')
    args = ap.parse_args()

    if args.nbar_cap and args.ideal_phonon:
        ap.error('--nbar-cap と --ideal-phonon は併用不可 (cap 0 が同義)')

    params = default_noise_params()
    tp = None
    overrides = {}
    if args.cool_us is not None:
        overrides['cool'] = args.cool_us
    if args.cool_residual is not None:
        overrides['phonon_cool_residual_axial'] = args.cool_residual
        overrides['phonon_cool_residual_transversal'] = args.cool_residual
        overrides['phonon_cool_residual_transversal2'] = args.cool_residual
    if args.params is not None or overrides:
        tp = load_time_parameters(args.params, overrides=overrides or None)

    arm_defs = {
        'no-cool': ('no-cool', {}),
        'anc': (f"anc@{args.ancilla_cool_us:g}us",
                {'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
        'merge': (f"merge@{args.merge_cool_threshold:g}+anc",
                  {'merge_cool_threshold_transversal': args.merge_cool_threshold,
                   'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
        'thr': (f"thr@{args.threshold:g}",
                {'cool_threshold_transversal': args.threshold}),
        'pred': (f"pred@{args.pred_threshold:g}x{args.pred_max_iters}+anc",
                 {'predictive_cool_threshold_transversal': args.pred_threshold,
                  'predictive_cool_max_iters': args.pred_max_iters,
                  'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
        'resident': (f"res@{args.resident_coolants}c+anc",
                     {'resident_coolant_ids': [9000 + i for i in
                                               range(args.resident_coolants)],
                      'coolant_recool_us': args.coolant_recool_us,
                      'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
        'resident-pred': (
            f"res@{args.resident_coolants}c+pred@{args.pred_threshold:g}"
            f"x{args.pred_max_iters}",
            {'resident_coolant_ids': [9000 + i for i in
                                      range(args.resident_coolants)],
             'coolant_recool_us': args.coolant_recool_us,
             'predictive_cool_threshold_transversal': args.pred_threshold,
             'predictive_cool_max_iters': args.pred_max_iters,
             'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
        'pred-data': (
            f"res@{args.resident_coolants}c+predD@{args.pred_threshold:g}",
            {'resident_coolant_ids': [9000 + i for i in
                                      range(args.resident_coolants)],
             'coolant_recool_us': args.coolant_recool_us,
             'predictive_cool_threshold_transversal': args.pred_threshold,
             'predictive_cool_actuator': 'data',
             'cool_ancilla_after_measure_us': args.ancilla_cool_us}),
    }
    arms = [arm_defs[a] for a in args.arms]
    if args.planner == 'v2':
        arms = [(label, {**kw, 'planner': 'v2',
                         'prefetch_depth': args.prefetch_depth,
                         'v2_sort_horizon': args.v2_sort_horizon,
                         'v2_sort_trigger': args.v2_sort_trigger})
                for label, kw in arms]

    print(f"E-C6 1-LIZ distance scan: d={args.distances}, rounds={args.rounds}, "
          f"shots={args.shots}, arms={[a[0] for a in arms]}")

    if args.isolate_recool:
        k = args.isolate_recool_iters
        suffix = "+iso" if k == 1 else f"+iso{k}"
        arms = [(f"{label}{suffix}",
                 {**kw, 'isolate_ancilla_recool': True,
                  'isolate_recool_iters': k})
                for label, kw in arms]
    if args.ideal_phonon:
        arms = [(f"{label}+n0", kw) for label, kw in arms]
    if args.cool_residual is not None:
        arms = [(f"{label}@r{args.cool_residual:g}", kw) for label, kw in arms]

    rows = []
    for label, kw in arms:
        print(f"\n[{label}]")
        for d in args.distances:
            if args.nbar_cap:
                pre = compile_rounds_d(d, kw, args.rounds, tp=tp)
                tag = 'cap' if args.nbar_cap_mode == 'clamp' else 'const'
                for cap in args.nbar_cap:
                    rows.append(run_point(
                        f"{label}@{tag}{cap:g}", d, kw, args.rounds,
                        args.shots, args.seed, params, tp,
                        warn_on_clip=not args.quiet_clip,
                        time_model=args.time_model,
                        nbar_cap=cap, nbar_cap_mode=args.nbar_cap_mode,
                        precompiled=pre))
            else:
                rows.append(run_point(label, d, kw, args.rounds, args.shots,
                                      args.seed, params, tp,
                                      warn_on_clip=not args.quiet_clip,
                                      time_model=args.time_model,
                                      ideal_phonon=args.ideal_phonon))

    # Λ の集計と要約
    print(f"\n{'='*78}")
    print("  summary (cap/avg = gate-time transversal n̄, p_L = over "
          f"{args.rounds} rounds, Λ = p_L(d)/p_L(d+2), 1σ CI)")
    lambdas = []
    for label in dict.fromkeys(r['label'] for r in rows):
        by_d = {r['d']: r for r in rows if r['label'] == label}
        line = f"  {label:12s}"
        for d in args.distances:
            r = by_d.get(d)
            if r is None:
                continue
            clip = "*" if r['n_clipped'] else ""
            line += (f" | d{d}: cap {r['cap_tr']:.1f} avg {r['avg_tr']:.1f} "
                     f"p_L {r['p_L']:.4f}{clip}")
        print(line)
        for d in args.distances:
            if d + 2 in by_d and d in by_d:
                lam = lambda_ratio(by_d[d], by_d[d + 2])
                if lam is None:
                    continue
                lo, hi = lam['ci_1sigma']
                verdict = ("✓ >1" if lo > 1.0 else
                           ("✗ <1" if hi < 1.0 else "~1 (不定)"))
                print(f"      Λ_{d}{d+2} = {lam['lambda']:.3f} "
                      f"[{lo:.3f}, {hi:.3f}]  {verdict}   "
                      f"(per-round basis {lam['lambda_per_round']:.3f})")
                lambdas.append({'label': label, 'd_from': d, 'd_to': d + 2, **lam})
    print("  * = LUT 上限 (n̄=20) 超過ゲートあり: p_L は下限値 (誤り過小評価)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = out / f"cooling_distance_scan_r{args.rounds}_{ts}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'rows': rows, 'lambdas': lambdas},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {path}")


if __name__ == '__main__':
    main()
