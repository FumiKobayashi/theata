"""Measure the theoretical parallelism ceiling for a multi-LIZ compile.

Decomposes wall-clock into four numbers so we can tell whether the
141ms scheduler result is limited by emission/scheduling (fixable) or by
the compiled stream's intrinsic dependency structure (needs better
placement/partition):

  sequential      = sum of all op durations (no parallelism)
  scheduler       = achieved parallel time (current Scheduler timesteps)
  critical_path   = best possible makespan FOR THIS op stream, i.e. the
                    longest resource-conflicting chain (infinite parallelism,
                    respects zone/ion conflicts). Lower bound the Scheduler
                    could reach with perfect scheduling of the same stream.
  per_liz_ideal   = max over LIZ of that LIZ's serial time (assumes LIZs run
                    fully in parallel, each LIZ serial internally)

Interpretation:
  scheduler - critical_path  = scheduling/emission slack left on the table
  critical_path - per_liz_ideal = cross-LIZ dependency forced by the stream
                                  (emission order / placement coupling)

([[project_per_liz_stream_design]] の理論限界測定)

Usage:
    python experiments/exp_critical_path.py --circuit d5_reuse2 --num-liz 2
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stabilizer_partitioner import StabilizerPartitioner, StabilizerPartition
from spatial_partitioner import partition_spatial_row, compute_initial_placement_3tier
from estimator import Estimator
from parallel_estimator import ParallelEstimator
from validator import ParallelValidator


def compile_multi_liz(circuit, num_liz, code_distance, phase_alternation=True,
                      memory_capacity=20):
    """Replicate exp_multi_liz_smoke wiring (spatial_row + 3-tier)."""
    mod = __import__(circuit)
    program = parse(mod.CIRCUIT)
    arch = QCCDArchitecture.chained_mlm(
        num_liz=num_liz, liz_capacity=4, memory_capacity=memory_capacity)
    liz_ids = [z.id for z in arch.get_liz_zones()]

    spatial = partition_spatial_row(
        program=program, ancilla_qubits=mod.ANCILLA_QUBITS,
        liz_zone_ids=liz_ids, code_distance=code_distance,
        phase_alternation=phase_alternation)
    program = spatial.new_program  # rewritten circuit

    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    # Build the compile partition from the spatial result so it includes any
    # ancillae the partition allocated for LIZs beyond the circuit's count.
    partition = StabilizerPartition(ancilla_to_liz={
        a: liz for liz, a in spatial.ancilla_assignments.items()})
    initial = compute_initial_placement_3tier(spatial, arch, mod.DATA_QUBITS)

    compiler = QCCDCompiler(arch, config=CompilerConfig(eviction_strategy="balanced"))
    result = compiler.compile(dag, initial, partition=partition)
    return result, arch, initial, liz_ids


def critical_path_us(flat_ops, arch):
    """Longest resource-conflicting weighted chain over the op stream.

    Resource = zones_used ∪ ions_used. Two ops that share any resource must
    be ordered; the flat_ops order is a valid serialization, so process
    left-to-right tracking the latest finisher per resource.
    """
    from ast_def import GateOp, MeasureOp, ResetOp
    est = Estimator()
    pv = ParallelValidator(arch)  # reuse its _zones_used / _ions_used
    last_finish = {}   # resource -> finish time of its latest user
    max_finish = 0.0
    for op in flat_ops:
        _, dur = est._get_operation_time(op)
        resources = pv._zones_used(op) | {("ion", i) for i in pv._ions_used(op)}
        # gates occupy their LIZ zone exclusively (only one gate per LIZ at a
        # time); _zones_used returns {} for gates, so add the annotated zone.
        if isinstance(op, (GateOp, MeasureOp, ResetOp)) and getattr(op, 'zone_id', -1) != -1:
            resources.add(op.zone_id)
        start = 0.0
        for r in resources:
            if r in last_finish and last_finish[r] > start:
                start = last_finish[r]
        finish = start + dur
        for r in resources:
            last_finish[r] = finish
        if finish > max_finish:
            max_finish = finish
    return max_finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuit', default='d5_reuse2')
    ap.add_argument('--num-liz', type=int, default=2)
    ap.add_argument('--code-distance', type=int, default=5)
    ap.add_argument('--memory-capacity', type=int, default=20)
    ap.add_argument('--no-phase-alt', action='store_true')
    args = ap.parse_args()

    result, arch, initial, liz_ids = compile_multi_liz(
        args.circuit, args.num_liz, args.code_distance,
        phase_alternation=not args.no_phase_alt,
        memory_capacity=args.memory_capacity)

    # sequential + scheduler(parallel)
    seq = Estimator().estimate(result.operations).total_time
    pe = ParallelEstimator()
    sched = pe.estimate(result.timesteps).total_time
    per_liz = pe.estimate_per_liz(result.timesteps, arch)

    cp = critical_path_us(result.flat_ops, arch)

    # validation sanity
    valid = ParallelValidator(arch).validate(result.timesteps, initial).is_valid

    def ms(x):
        return x / 1000.0

    print("=" * 70)
    print(f"Critical-path decomposition: {args.circuit} num_liz={args.num_liz}")
    print(f"  LIZ zones: {liz_ids}   timesteps: {len(result.timesteps)}   valid: {valid}")
    print("=" * 70)
    print(f"  sequential     : {ms(seq):8.2f} ms   (sum of all op durations)")
    print(f"  scheduler      : {ms(sched):8.2f} ms   (achieved parallel)")
    print(f"  critical_path  : {ms(cp):8.2f} ms   (best possible for THIS stream)")
    print(f"  per_liz_ideal  : {ms(per_liz.total_time_max):8.2f} ms   (max over LIZ, LIZs parallel)")
    print("-" * 70)
    print(f"  scheduler - critical_path      = {ms(sched - cp):7.2f} ms"
          f"  (scheduling/emission slack)")
    print(f"  critical_path - per_liz_ideal  = {ms(cp - per_liz.total_time_max):7.2f} ms"
          f"  (cross-LIZ dependency in stream)")
    print("-" * 70)
    for liz, t in sorted(per_liz.per_liz_time.items()):
        n = per_liz.per_liz_op_count.get(liz, 0)
        print(f"    LIZ {liz}: {ms(t):7.2f} ms  ({n} ops)")


if __name__ == '__main__':
    main()
