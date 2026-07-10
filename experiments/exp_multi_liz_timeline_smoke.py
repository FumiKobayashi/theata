"""Multi-LIZ per-ion timeline smoke test.

実サイズ (d5 2-LIZ, boustrophedon + 3tier_usage) の per-LIZ stream コンパイル
結果から per-ion タイムラインを構築し、内蔵 assert を実データで通す:

  - 同一イオンの busy 区間非重複 (race-freedom の実測確認)
  - 各イオンの時刻順 op 列と merged_flat 出現順の一致 (cross-LIZ 順序検証)

さらにギャップ統計と timeline.json (可視化用) を出力する。

Usage:
    python experiments/exp_multi_liz_timeline_smoke.py --num-liz 2
"""

import sys
import json
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from qccd_compiler import QCCDArchitecture, CompilerConfig
from per_liz_compiler import compile_per_liz, estimate_streams
from ion_timeline import (
    gaps_from_per_liz_results, op_intervals_from_streams,
    build_ion_events, timeline_to_dict,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuit', default='d5_reuse2')
    ap.add_argument('--num-liz', type=int, default=2)
    ap.add_argument('--code-distance', type=int, default=5)
    ap.add_argument('--stab-order', default='boustrophedon')
    ap.add_argument('--placement', default='3tier_usage')
    ap.add_argument('--out-dir', default='results/per_ion_noise')
    args = ap.parse_args()

    mod = __import__(args.circuit)
    program = parse(mod.CIRCUIT)
    arch = QCCDArchitecture.chained_mlm(
        num_liz=args.num_liz, liz_capacity=4, memory_capacity=20)

    print(f"Compiling {args.circuit} ({args.num_liz}-LIZ, "
          f"{args.stab_order} + {args.placement})...")
    result = compile_per_liz(
        program=program, ancilla_qubits=mod.ANCILLA_QUBITS,
        data_qubits=mod.DATA_QUBITS, arch=arch,
        code_distance=args.code_distance,
        config=CompilerConfig(eviction_strategy="balanced"),
        stab_order=args.stab_order, placement_mode=args.placement)
    estimate_streams(result)
    print(f"  wall clock: {result.wall_clock_us/1000:.2f} ms, "
          f"segments: {len(result.segments)}")

    # --- タイムライン構築 (assert はこの中で実行される) ---
    anchor_ops, gaps = gaps_from_per_liz_results([result])
    print(f"\n[asserts] busy 非重複 + merged_flat 順序一致: PASS")

    intervals, t_end = op_intervals_from_streams(result)
    events = build_ion_events(intervals)

    n_ions = len(events)
    all_gaps = [(ion, g) for gd in gaps.gaps_by_op.values()
                for ion, g in gd.items()]
    gap_vals = [g for _, g in all_gaps]
    cross_core_ions = sorted(
        ion for ion, evs in events.items()
        if len({ev.core for ev in evs}) > 1)

    print(f"\n[timeline stats]")
    print(f"  ions: {n_ions}, anchors: {len(anchor_ops)}, "
          f"nonzero gaps: {len(gap_vals)}")
    print(f"  gap (μs): min={min(gap_vals):.1f}  max={max(gap_vals):.1f}  "
          f"mean={sum(gap_vals)/len(gap_vals):.1f}")
    print(f"  total idle exposure: "
          f"{(sum(gap_vals) + sum(gaps.trailing_idle.values()))/1000:.1f} qubit·ms")
    print(f"  cross-core ions (両 LIZ の anchor に登場): {len(cross_core_ions)} 個")
    print(f"    {cross_core_ions}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tl_path = out / (f"{args.circuit}_{args.num_liz}liz_timeline.json")
    tl_path.write_text(
        json.dumps({'wall_clock_us': t_end,
                    'ions': timeline_to_dict(events)}, indent=1),
        encoding='utf-8')
    print(f"\n  saved: {tl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
