"""Multi-LIZ smoke test.

最小構成で 2-LIZ + d5_reuse2 + ancilla_locked partition を試す。
目的: 多 LIZ 対応の compiler が動くか確認、失敗パターンを観測する。

使い方:
    python experiments/exp_multi_liz_smoke.py
    python experiments/exp_multi_liz_smoke.py --circuit d5_reuse2
    python experiments/exp_multi_liz_smoke.py --circuit d7_reuse2
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stabilizer_partitioner import StabilizerPartitioner, StabilizerPartition
from spatial_partitioner import partition_spatial_row, compute_initial_placement_3tier
from parallel_estimator import ParallelEstimator, save_parallel_txt
from validator import validate_operations, validate_timesteps


def build_initial_placement(data_qubits, ancilla_qubits, num_liz, arch):
    """Distribute qubits across memory zones.

    Strategy: ancilla i -> memory zone closest to LIZ i.
    Data qubits split evenly across all memory zones.
    """
    # Memory zones in order from left to right
    memory_zones = [z.id for z in arch.zones if z.zone_type == "M"]
    liz_zones = [z.id for z in arch.zones if z.zone_type == "L"]

    initial = {}

    # Ancillae: each ancilla in the memory zone immediately to the left of its LIZ
    for i, ancilla in enumerate(ancilla_qubits):
        if i < len(liz_zones):
            liz_id = liz_zones[i]
            # Memory zone immediately left of this LIZ (zone_id = liz_id - 2)
            mem_id = liz_id - 2
            initial[ancilla] = mem_id
        else:
            # Fallback: distribute round-robin
            initial[ancilla] = memory_zones[i % len(memory_zones)]

    # Data qubits: split across memory zones
    n_data = len(data_qubits)
    n_mem = len(memory_zones)
    chunk = (n_data + n_mem - 1) // n_mem
    for i, qid in enumerate(data_qubits):
        zone_idx = min(i // chunk, n_mem - 1)
        initial[qid] = memory_zones[zone_idx]

    return initial


def main():
    parser = argparse.ArgumentParser(description='Multi-LIZ smoke test')
    parser.add_argument('--circuit', default='d5_reuse2',
                        help='Circuit module name (default: d5_reuse2)')
    parser.add_argument('--num-liz', type=int, default=2,
                        help='Number of LIZs (default: 2)')
    parser.add_argument('--liz-capacity', type=int, default=4)
    parser.add_argument('--memory-capacity', type=int, default=20)
    parser.add_argument('--stabilizer-parallel', action='store_true',
                        help='Use stabilizer_parallel=True in DAGBuilder')
    parser.add_argument('--partition', default='ancilla_locked',
                        choices=['ancilla_locked', 'spatial_row'],
                        help='Partition strategy (default: ancilla_locked)')
    parser.add_argument('--code-distance', type=int, default=5,
                        help='Code distance for spatial partition (default: 5)')
    parser.add_argument('--phase-alternation', action='store_true',
                        help='Reorder per-LIZ chains to anti-correlate central-M traffic')
    parser.add_argument('--dump-timesteps', metavar='PATH',
                        help='Save TimeStep view to a text file for inspection')
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-LIZ Smoke Test")
    print(f"  circuit: {args.circuit}")
    print(f"  num_liz: {args.num_liz}")
    print(f"  liz_capacity: {args.liz_capacity}")
    print(f"  memory_capacity: {args.memory_capacity}")
    print("=" * 70)

    # 1. Architecture
    arch = QCCDArchitecture.chained_mlm(
        num_liz=args.num_liz,
        liz_capacity=args.liz_capacity,
        memory_capacity=args.memory_capacity,
        temp_capacity=args.memory_capacity,
    )
    print(f"\nArchitecture: {arch}")
    liz_ids = [z.id for z in arch.zones if z.zone_type == "L"]
    print(f"LIZ zone IDs: {liz_ids}")

    # 2. Circuit
    mod = __import__(args.circuit)
    program = parse(mod.CIRCUIT)
    print(f"\nstabilizer_parallel: {args.stabilizer_parallel}")
    print(f"partition strategy: {args.partition}")

    # 2b. Spatial partition (rewrite circuit) if requested
    spatial_result = None
    if args.partition == 'spatial_row':
        spatial_result = partition_spatial_row(
            program=program,
            ancilla_qubits=mod.ANCILLA_QUBITS,
            liz_zone_ids=liz_ids,
            code_distance=args.code_distance,
            phase_alternation=args.phase_alternation,
        )
        if args.phase_alternation:
            print(f"  phase_alternation: ON (per-LIZ chain reordered)")
        program = spatial_result.new_program
        print(f"\nSpatial partition (row-band):")
        print(f"  stabilizer count: {len(spatial_result.stabilizer_to_liz)}")
        for liz in liz_ids:
            n = sum(1 for v in spatial_result.stabilizer_to_liz.values() if v == liz)
            print(f"    LIZ {liz}: {n} stabilizers")
        print(f"  pure data qubits: {len(spatial_result.pure_data)}")
        print(f"  shared data qubits: {len(spatial_result.shared_data)}  -> {sorted(spatial_result.shared_data)}")

    dag = DAGBuilder(program, stabilizer_parallel=args.stabilizer_parallel).build()
    print(f"\nCircuit: {args.circuit}")
    print(f"  data qubits: {len(mod.DATA_QUBITS)}")
    print(f"  ancilla qubits: {mod.ANCILLA_QUBITS}")
    print(f"  DAG nodes: {len(dag.nodes)}")

    # 3. Partition
    if spatial_result is not None:
        # Build from the spatial result so it includes ancillae allocated for
        # LIZs beyond the circuit's ancilla count (N-LIZ).
        partition = StabilizerPartition(ancilla_to_liz={
            a: liz for liz, a in spatial_result.ancilla_assignments.items()})
    else:
        partitioner = StabilizerPartitioner(strategy="ancilla_locked")
        partition = partitioner.partition(mod.ANCILLA_QUBITS, liz_ids)
    print(f"\nPartition (ancilla -> LIZ): {partition.ancilla_to_liz}")

    # 4. Initial placement
    if spatial_result is not None:
        initial = compute_initial_placement_3tier(
            spatial_result, arch, mod.DATA_QUBITS,
        )
        print(f"\nInitial placement (3-tier from spatial partition):")
        for ancilla in mod.ANCILLA_QUBITS:
            if ancilla in initial:
                print(f"  ancilla {ancilla} -> zone {initial[ancilla]}")
        # Summary by zone
        from collections import defaultdict
        zone_counts = defaultdict(int)
        for q, z in initial.items():
            zone_counts[z] += 1
        print(f"  zone occupancy: {dict(sorted(zone_counts.items()))}")
    else:
        initial = build_initial_placement(
            mod.DATA_QUBITS, mod.ANCILLA_QUBITS, args.num_liz, arch
        )
        print(f"\nInitial placement (legacy round-robin):")
        for ancilla in mod.ANCILLA_QUBITS:
            print(f"  ancilla {ancilla} -> zone {initial[ancilla]}")
        print(f"  (data qubits distributed across zones)")

    # 5. Compile
    config = CompilerConfig(eviction_strategy="balanced")
    compiler = QCCDCompiler(arch, config=config)

    print("\n" + "=" * 70)
    print("Compiling...")
    print("=" * 70)
    try:
        result = compiler.compile(dag, initial, partition=partition)
        print(f"\n[OK] Compile succeeded")
        print(f"  timesteps: {len(result.timesteps)}")
        print(f"  total ops: {result.stats.get('total_ops', '?')}")
    except Exception as e:
        print(f"\n[FAIL] Compile raised: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 6. Estimate
    est = ParallelEstimator()
    par_result = est.estimate(result.timesteps)
    print(f"\nExecution time (Scheduler-based parallel):")
    print(f"  parallel:   {par_result.total_time/1000:.2f} ms")
    print(f"  sequential: {par_result.sequential_time/1000:.2f} ms")
    print(f"  speedup:    {par_result.speedup:.2f}x")

    # Per-LIZ analysis
    per_liz = est.estimate_per_liz(result.timesteps, arch)
    print(f"\nPer-LIZ time analysis:")
    print(per_liz)

    # Optional: dump TimeStep view to a file for inspection
    if args.dump_timesteps:
        save_parallel_txt(
            result.timesteps,
            args.dump_timesteps,
            initial_placement=initial,
            arch=arch,
            estimator=est,
        )

    # 7. Validate
    print("\n" + "=" * 70)
    print("Validating...")
    print("=" * 70)
    flat_ops = [op for ts in result.timesteps for op in ts.ops]
    op_result = validate_operations(flat_ops, initial, arch)
    ts_result = validate_timesteps(result.timesteps, initial, arch)

    def report(label, vresult):
        errors = vresult.errors
        if not errors:
            print(f"\n[OK] {label} validation passed")
            return True
        print(f"\n[FAIL] {label} validation: {len(errors)} errors")
        # group by error code
        from collections import Counter
        codes = Counter(getattr(e, 'code', str(e)) for e in errors)
        print(f"  error categories: {dict(codes)}")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        return False

    op_ok = report("Operation", op_result)
    ts_ok = report("Timestep", ts_result)
    return 0 if (op_ok and ts_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
