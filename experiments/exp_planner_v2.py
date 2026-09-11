"""planner v2 (per-zone state machine + JIT staging) の検証実験。

v1 / v2(prefetch_depth=0) / v2(prefetch_depth=1) を同一回路・同一配置で
コンパイルし、以下を比較する:

  - T_lock (ロックステップ wall) / T_event (イベント駆動 wall)
  - 命令並列度 (flat_ops / timesteps)
  - 搬送 op 数 (shuttle / swap / split / merge)
  - ゲート時 n̄ (cap / avg) — 整列 swap の加熱コストを含む実会計
  - 測定窓に隠れた搬送量 (hidden_us)
  - 検証: per-ion busy 非重複 + linearization 順一致 (両スケジューラ)

rounds >= 2 では final placement / zone_contents を持ち越して同一回路を
繰り返す (exp_verify2_general と同じ方式、reverse なし簡易版)。

Usage:
    python experiments/exp_planner_v2.py                    # d3+d5, 3 rounds
    python experiments/exp_planner_v2.py d5_reuse1 --rounds 1
    python experiments/exp_planner_v2.py --horizon 6
"""

import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator
from event_scheduler import schedule_events
from ion_timeline import (
    op_intervals_from_timesteps, build_ion_events, validate_order_against,
)
from exp_multi_round import placement_greedy_alt
from exp_event_schedule import measure_hiding_stats
from exp_distance_scan import DISTANCE_CIRCUITS

DEFAULT_DISTANCES = [3, 5]


def run_variant(d, config, rounds):
    """rounds 回 compile し (fwd/rev 交互 + 配置持ち越し = 正準手法)、集計を返す。"""
    forward, reverse, _ = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    pe = ParallelEstimator()
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)
    placement, zone_contents = initial, None
    tot = {'shuttles': 0, 'swaps': 0, 'splits': 0, 'merges': 0}
    lock_ms, event_ms, gate_tr = [], [], []
    n_ts = n_flat = 0
    hidden_us = 0.0
    for r in range(1, rounds + 1):
        mod = __import__(forward if r % 2 == 1 else reverse)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=config).compile(dag, placement)
        for k in tot:
            tot[k] += cr.stats.get(k, 0)
        lock_ms.append(pe.estimate(cr.timesteps).total_time / 1000)
        intervals, t_end = schedule_events(cr.flat_ops)
        event_ms.append(t_end / 1000)
        n_ts += len(cr.timesteps)
        n_flat += len(cr.flat_ops)
        # 検証: 両スケジューラで busy 非重複 + linearization 順一致
        lock_iv, _ = op_intervals_from_timesteps(cr.timesteps)
        validate_order_against(build_ion_events(lock_iv), cr.flat_ops)
        validate_order_against(build_ion_events(intervals), cr.flat_ops)
        _, _, h = measure_hiding_stats(intervals)
        hidden_us += h
        for g in (cr.phonon_data or {}).get('at_gates', []):
            gate_tr.append(max(p[1] for p in g['phonons'].values()))
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
    return {
        'transport': tot,
        'lock_ms': lock_ms, 'T_lock_ms': sum(lock_ms),
        'event_ms': event_ms, 'T_event_ms': sum(event_ms),
        'flat_ops': n_flat, 'timesteps': n_ts,
        'parallelism': n_flat / n_ts if n_ts else 0.0,
        'hidden_us': hidden_us,
        'nbar_cap': max(gate_tr) if gate_tr else 0.0,
        'nbar_avg': sum(gate_tr) / len(gate_tr) if gate_tr else 0.0,
    }


def variants(horizons):
    """v1 / 測定窓整列 (現行 v2) / 納品時真端整列 (v3 案) / 併用。"""
    out = [('v1', CompilerConfig())]
    for h in horizons:
        out.append((f'meas h{h}',
                    CompilerConfig(planner='v2', prefetch_depth=0,
                                   v2_sort_horizon=h,
                                   v2_sort_trigger='measure')))
    out.append(('deliv h1',
                CompilerConfig(planner='v2', prefetch_depth=0,
                               v2_sort_horizon=1,
                               v2_sort_trigger='delivery')))
    out.append(('deliv h2',
                CompilerConfig(planner='v2', prefetch_depth=0,
                               v2_sort_horizon=2,
                               v2_sort_trigger='delivery')))
    out.append(('both h2',
                CompilerConfig(planner='v2', prefetch_depth=0,
                               v2_sort_horizon=2,
                               v2_sort_trigger='both')))
    return out


def csv_row(d, label, rounds, cfg, r):
    """placement_bridge / verify2 と同じ流儀の 1 行 (utf-8-sig CSV 用)。"""
    t = r['transport']
    return {
        'd': d, 'variant': label, 'rounds': rounds,
        'planner': cfg.planner,
        'sort_trigger': cfg.v2_sort_trigger if cfg.planner == 'v2' else '',
        'sort_horizon': cfg.v2_sort_horizon if cfg.planner == 'v2' else '',
        'prefetch_depth': cfg.prefetch_depth if cfg.planner == 'v2' else '',
        'n_shuttle': t['shuttles'], 'n_swap': t['swaps'],
        'n_split': t['splits'], 'n_merge': t['merges'],
        'T_lock_ms': round(r['T_lock_ms'], 1),
        'T_event_ms': round(r['T_event_ms'], 1),
        'round_lock_ms': ';'.join(f"{x:.1f}" for x in r['lock_ms']),
        'round_event_ms': ';'.join(f"{x:.1f}" for x in r['event_ms']),
        'flat_ops': r['flat_ops'], 'timesteps': r['timesteps'],
        'parallelism': round(r['parallelism'], 3),
        'hidden_us': round(r['hidden_us'], 0),
        'nbar_cap': round(r['nbar_cap'], 2),
        'nbar_avg': round(r['nbar_avg'], 2),
    }


def main():
    ap = argparse.ArgumentParser(description='planner v2 comparison')
    ap.add_argument('--distances', type=int, nargs='*', default=DEFAULT_DISTANCES)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--horizons', type=int, nargs='*', default=[2],
                    help='v2 の sort horizon (複数指定で CSV に並ぶ)')
    args = ap.parse_args()

    all_results = []
    csv_rows = []
    for d in args.distances:
        name = f"d={d}"
        print(f"\n=== {name} (rounds={args.rounds}, horizons={args.horizons}) ===")
        print(f"  {'variant':8s} | {'T_lock':>8s} {'T_evt':>8s} | {'par':>5s} | "
              f"{'shut':>5s} {'swap':>5s} {'split':>6s} {'merge':>6s} | "
              f"{'cap':>6s} {'avg':>5s} | {'hidden':>7s}")
        for label, cfg in variants(args.horizons):
            r = run_variant(d, cfg, args.rounds)
            t = r['transport']
            print(f"  {label:9s}| {r['T_lock_ms']:8.1f} {r['T_event_ms']:8.1f} | "
                  f"{r['parallelism']:5.2f} | {t['shuttles']:5d} {t['swaps']:5d} "
                  f"{t['splits']:6d} {t['merges']:6d} | {r['nbar_cap']:6.2f} "
                  f"{r['nbar_avg']:5.2f} | {r['hidden_us']/1000:6.1f}ms",
                  flush=True)
            all_results.append({'circuit': name, 'variant': label,
                                'rounds': args.rounds, **r})
            csv_rows.append(csv_row(d, label, args.rounds, cfg, r))

    out_dir = Path(__file__).parent.parent / 'results' / 'planner_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = out_dir / f'planner_v2_{ts}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'planner_v2_comparison',
                   'generated_at': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, ensure_ascii=False)
    out_csv = out_dir / f'planner_v2_{ts}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved: {out}")
    print(f"       {out_csv}")


if __name__ == '__main__':
    main()
