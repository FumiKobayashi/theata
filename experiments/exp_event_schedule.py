"""Event-driven scheduling experiment.

同じ flat_ops に対して、既存のロックステップ timestep モデルと
イベント駆動 (ゾーン空き時刻) モデルの時間を比較する。

検証項目:
  1. 整合性: per-ion busy 区間の非重複 + linearization 順との一致
     (ion_timeline の assert がそのまま検証になる)
  2. wall clock の短縮量
  3. Measure の陰に隠れた搬送 op の数と時間 (隠蔽効果の直接計測)
  4. anchor gap (dephasing として Z_ERROR になる per-ion idle) の合計変化

使い方:
    python experiments/exp_event_schedule.py                # デフォルト回路セット
    python experiments/exp_event_schedule.py d5_reuse1      # 特定回路のみ
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from ast_def import MeasureOp, ShuttleOp, SplitOp, MergeOp, SwapOp, CoolOp
from parallel_estimator import ParallelEstimator
from event_scheduler import schedule_events
from ion_timeline import (
    op_intervals_from_timesteps, build_ion_events,
    validate_order_against, compute_ion_gaps,
)

DEFAULT_CIRCUITS = ["snake_reuse1", "d5_reuse1"]
TRANSPORT_TYPES = (ShuttleOp, SplitOp, MergeOp, SwapOp)


def compile_circuit(name: str, placement: str = "greedy_alt"):
    mod = __import__(name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    half = (len(mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    if placement == "greedy_alt":
        # 正準配置 (exp_distance_scan / exp_build_examples と同じ)
        from exp_multi_round import placement_greedy_alt
        initial = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
    else:
        # 素朴な index 順 2 分割 (対照用)
        initial = {}
        for i, q in enumerate(mod.DATA_QUBITS):
            initial[q] = 1 if i < len(mod.DATA_QUBITS) // 2 else 5
        for a in mod.ANCILLA_QUBITS:
            initial[a] = 5
    result = QCCDCompiler(arch, config=CompilerConfig()).compile(dag, initial)
    return result


def measure_hiding_stats(intervals):
    """各 Measure 区間に (部分的にでも) 重なった搬送 op の数と重なり時間を集計。"""
    meas = [iv for iv in intervals if isinstance(iv.op, MeasureOp)]
    n_hidden = 0
    hidden_us = 0.0
    for m in meas:
        for iv in intervals:
            if iv is m or not isinstance(iv.op, TRANSPORT_TYPES):
                continue
            ov = min(iv.t1, m.t1) - max(iv.t0, m.t0)
            if ov > 1e-9:
                n_hidden += 1
                hidden_us += ov
    return len(meas), n_hidden, hidden_us


def total_anchor_gap(intervals, t_end):
    """anchor gap (Z_ERROR になる per-ion idle) の全イオン合計。"""
    events = build_ion_events(intervals)   # busy 非重複 assert 込み
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=t_end)
    return sum(g for per_ion in gaps.gaps_by_op.values() for g in per_ion.values())


def run_one(name: str, placement: str = "greedy_alt"):
    print(f"\n=== {name} (placement={placement}) ===")
    result = compile_circuit(name, placement)
    flat_ops = result.flat_ops

    # --- lockstep (既存モデル) ---
    lock_intervals, lock_end = op_intervals_from_timesteps(result.timesteps)
    lock_events = build_ion_events(lock_intervals)
    validate_order_against(lock_events, flat_ops)
    lock_gap = total_anchor_gap(lock_intervals, lock_end)

    # --- event-driven ---
    ev_intervals, ev_end = schedule_events(flat_ops)
    ev_events = build_ion_events(ev_intervals)      # busy 非重複 assert
    validate_order_against(ev_events, flat_ops)     # linearization 順 assert
    ev_gap = total_anchor_gap(ev_intervals, ev_end)

    n_meas, n_hidden, hidden_us = measure_hiding_stats(ev_intervals)
    _, n_hidden_lock, hidden_lock_us = measure_hiding_stats(lock_intervals)

    print(f"  [OK] validation passed (busy non-overlap + linearization order)")
    print(f"  wall clock : lockstep {lock_end/1000:.2f} ms -> event {ev_end/1000:.2f} ms "
          f"({100*(1 - ev_end/lock_end):.1f}% down)")
    print(f"  anchor gap : lockstep {lock_gap/1000:.2f} ms -> event {ev_gap/1000:.2f} ms "
          f"(sum over ions; dephasing に入る idle)")
    print(f"  hidden transport under Measure: lockstep {n_hidden_lock} ops "
          f"({hidden_lock_us:.0f} us) -> event {n_hidden} ops ({hidden_us:.0f} us) "
          f"/ {n_meas} measures")

    return {
        "circuit": name,
        "placement": placement,
        "flat_ops": len(flat_ops),
        "lockstep": {
            "wall_clock_us": lock_end,
            "anchor_gap_sum_us": lock_gap,
            "hidden_transport_ops": n_hidden_lock,
            "hidden_transport_us": hidden_lock_us,
        },
        "event": {
            "wall_clock_us": ev_end,
            "anchor_gap_sum_us": ev_gap,
            "hidden_transport_ops": n_hidden,
            "hidden_transport_us": hidden_us,
        },
        "num_measures": n_meas,
    }


def main():
    parser = argparse.ArgumentParser(description='Event-driven scheduling experiment')
    parser.add_argument('circuits', nargs='*', default=DEFAULT_CIRCUITS)
    parser.add_argument('--placement', choices=['greedy_alt', 'naive'],
                        default='greedy_alt',
                        help='greedy_alt = 正準配置 (exp_distance_scan と同じ), '
                             'naive = index 順 2 分割')
    args = parser.parse_args()

    results = [run_one(name, args.placement) for name in args.circuits]

    out_dir = Path(__file__).parent.parent / 'results' / 'event_schedule'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"event_schedule_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "event_schedule_comparison",
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
