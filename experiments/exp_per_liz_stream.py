"""Per-LIZ stream compile の時間 / フォノン測定。

現行 (global TimeStep) コンパイルと per-LIZ stream コンパイルを同じ回路・
同じアーキテクチャで比較する:

  - 現行 scheduler 時間 (global TimeStep 列)
  - per-LIZ stream の wall clock (event simulation)
  - per-core busy 時間 / sync 待ち
  - 参考: per_liz_ideal (完全並列の楽観下限), 1-LIZ 実行時間
  - フォノン最大値 (axial / transversal)
  - validation (併合ビューを ParallelValidator で検証)

Usage:
    python experiments/exp_per_liz_stream.py --circuit d5_reuse2 --num-liz 2
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from qccd_parser import parse
from qccd_compiler import QCCDArchitecture, CompilerConfig
from parallel_estimator import ParallelEstimator
from per_liz_compiler import (
    compile_per_liz, estimate_streams, validate_merged, save_streams,
)
from exp_critical_path import compile_multi_liz, critical_path_us


def ms(x):
    return x / 1000.0


def run_baseline(circuit, num_liz, code_distance, memory_capacity):
    """現行 global-TimeStep コンパイル。"""
    result, arch, initial, liz_ids = compile_multi_liz(
        circuit, num_liz, code_distance, phase_alternation=True,
        memory_capacity=memory_capacity)
    pe = ParallelEstimator()
    sched = pe.estimate(result.timesteps).total_time
    per_liz = pe.estimate_per_liz(result.timesteps, arch)
    cp = critical_path_us(result.flat_ops, arch)
    ph = result.phonon_data or {}
    return {
        'sched_us': sched,
        'per_liz_ideal_us': per_liz.total_time_max,
        'critical_path_us': cp,
        'phonon': (ph.get('max_axial', 0.0), ph.get('max_transversal', 0.0)),
        'timesteps': len(result.timesteps),
    }


def run_single_liz(circuit, code_distance):
    """参考: 1-LIZ 実行時間 (canonical: exp_build_examples と同一 wiring)。

    注意: compile_multi_liz(num_liz=1) は使わない。3-tier placement は
    num_liz=1 で全 data を単一 Memory に詰める退化配置になり、時間もフォノンも
    大きく水増しされる (examples/single_liz の greedy_alt 配置が正)。
    """
    try:
        from exp_build_examples import (
            DISTANCE_CONFIG, make_arch, initial_placement, compile_one,
        )
        fwd_name = DISTANCE_CONFIG[code_distance][0]
        fwd = __import__(fwd_name)
        arch = make_arch(fwd)
        placement = initial_placement(fwd)
        _, result = compile_one(fwd_name, placement, arch)
        pe = ParallelEstimator()
        ph = result.phonon_data or {}
        return {
            'sched_us': pe.estimate(result.timesteps).total_time,
            'phonon': (ph.get('max_axial', 0.0), ph.get('max_transversal', 0.0)),
            'circuit': fwd_name,
        }
    except Exception as exc:
        print(f"  (1-LIZ canonical baseline skipped: {exc})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuit', default='d5_reuse2')
    ap.add_argument('--num-liz', type=int, default=2)
    ap.add_argument('--code-distance', type=int, default=5)
    ap.add_argument('--memory-capacity', type=int, default=20)
    ap.add_argument('--skip-single', action='store_true')
    ap.add_argument('--out-dir', default=None,
                    help='保存先 (default: results/per_liz_stream)')
    ap.add_argument('--stab-order', default='phase_alt',
                    choices=['phase_alt', 'boustrophedon'])
    ap.add_argument('--placement', default='3tier',
                    choices=['3tier', 'greedy_alt', '3tier_usage'])
    args = ap.parse_args()

    mod = __import__(args.circuit)
    program = parse(mod.CIRCUIT)
    arch = QCCDArchitecture.chained_mlm(
        num_liz=args.num_liz, liz_capacity=4,
        memory_capacity=args.memory_capacity)

    # --- per-LIZ stream compile ---
    result = compile_per_liz(
        program=program, ancilla_qubits=mod.ANCILLA_QUBITS,
        data_qubits=mod.DATA_QUBITS, arch=arch,
        code_distance=args.code_distance,
        config=CompilerConfig(eviction_strategy="balanced"),
        stab_order=args.stab_order, placement_mode=args.placement)
    estimate_streams(result)
    validation = validate_merged(result)
    phonon = result.phonon_summary()

    # --- baselines ---
    baseline = run_baseline(args.circuit, args.num_liz, args.code_distance,
                            args.memory_capacity)
    single = None if args.skip_single else run_single_liz(
        args.circuit, args.code_distance)

    print("=" * 72)
    print(f"Per-LIZ stream: {args.circuit}  num_liz={args.num_liz}")
    print("=" * 72)

    print(f"\n[blocks] (segment 集計, 全 {len(result.segments)} segments, "
          f"sync 辺 {len(result.syncs)} 本)")
    for rec in result.epoch_summary():
        print(f"  core={rec['core_id']:2d} {rec['label']:8s} "
              f"segs={rec['segments']:4d} ops={rec['ops']:5d}  "
              f"busy={ms(rec['busy_us']):7.2f} ms  "
              f"[{ms(rec['start_us']):7.2f} → {ms(rec['end_us']):7.2f}]")

    binding = sorted((s for s in result.syncs if s.wait_us > 0),
                     key=lambda s: -s.wait_us)
    total_wait = sum(s.wait_us for s in binding)
    print(f"\n[sync waits] 実際に待ちを生んだ辺: {len(binding)} 本, "
          f"合計 {ms(total_wait):.2f} ms")
    for s in binding[:8]:
        si_, sj_ = result.segments[s.src], result.segments[s.dst]
        print(f"  seg{s.src}(core{si_.core_id}.{si_.label}) → "
              f"seg{s.dst}(core{sj_.core_id}.{sj_.label})  "
              f"zones={sorted(s.shared_zones)} "
              f"ions={sorted(s.shared_ions)[:6]}  wait={ms(s.wait_us):.2f} ms")

    print("\n[per-core timeline]")
    for core, idxs in sorted(result.streams.items()):
        busy = result.per_core_busy_us[core]
        end = result.per_core_end_us[core]
        print(f"  core {core}: busy={ms(busy):7.2f} ms  end={ms(end):7.2f} ms  "
              f"idle={ms(end - busy):6.2f} ms")

    print("\n[comparison]")
    if single:
        print(f"  1-LIZ scheduler        : {ms(single['sched_us']):8.2f} ms"
              f"   (canonical greedy_alt, {single['circuit']})")
    print(f"  {args.num_liz}-LIZ scheduler (現行) : "
          f"{ms(baseline['sched_us']):8.2f} ms   "
          f"({baseline['timesteps']} timesteps)")
    print(f"  {args.num_liz}-LIZ critical_path   : "
          f"{ms(baseline['critical_path_us']):8.2f} ms   (旧 stream の下限)")
    print(f"  {args.num_liz}-LIZ per_liz_ideal   : "
          f"{ms(baseline['per_liz_ideal_us']):8.2f} ms   (完全並列の楽観下限)")
    print(f"  {args.num_liz}-LIZ per-LIZ streams : "
          f"{ms(result.wall_clock_us):8.2f} ms   ← 本実装 (wall clock)")

    print("\n[phonon max (axial / transversal)]")
    if single:
        sx, st_ = single['phonon']
        print(f"  1-LIZ    : {sx:8.2f} / {st_:6.3f}")
    bx, bt = baseline['phonon']
    print(f"  現行     : {bx:8.2f} / {bt:6.3f}")
    print(f"  per-LIZ  : {phonon['max_axial']:8.2f} / "
          f"{phonon['max_transversal']:6.3f}")

    print(f"\n[validation] merged view: "
          f"{'PASS' if validation.is_valid else 'FAIL'}"
          f"  ({len(validation.errors)} errors)")
    if not validation.is_valid:
        for err in validation.errors[:10]:
            print(f"    {err}")

    out_dir = args.out_dir or str(
        Path(__file__).parent.parent / 'results' / 'per_liz_stream')
    prefix = f"{args.circuit}_{args.num_liz}liz"
    if args.stab_order != 'phase_alt':
        prefix += f"_{args.stab_order}"
    if args.placement != '3tier':
        prefix += f"_{args.placement}"
    written = save_streams(result, out_dir, prefix)
    print("\n[saved]")
    for p in written:
        print(f"  {p}")


if __name__ == '__main__':
    main()
