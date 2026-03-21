"""
Stabilizer Interleave Experiment

2スタビライザ間のCNOT interleaveとMeasureバッチ化の効果を検証する。
以下の構成を比較:
  1. baseline:        sequential, batch=1
  2. interleave:      interleave, batch=1
  3. interleave+batch: interleave, batch=2

使い方:
    python exp_interleave.py
    python exp_interleave.py snake_per_stabilizer
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from estimator import Estimator
from parallel_estimator import ParallelEstimator, save_parallel_txt
from validator import validate_operations, validate_timesteps
from ast_def import GateOp, MeasureOp

# ==========================================
# 設定
# ==========================================

CIRCUIT_MODULES = [
    "snake_per_stabilizer",
]

LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20
TEMP_CAPACITY = 20


# ==========================================
# 実験ロジック
# ==========================================

def compile_circuit(circuit_module_name: str, interleave: bool, batch_size: int,
                    memory_reorder: bool = False, initial_override: dict = None):
    """回路をコンパイルして結果を返す"""
    circuit_module = importlib.import_module(circuit_module_name)

    program = parse(circuit_module.CIRCUIT)

    if interleave:
        builder = DAGBuilder(program, stabilizer_parallel=True)
        dag = builder.build()
        builder.add_hook_error_edges()
    else:
        builder = DAGBuilder(program, stabilizer_parallel=False)
        dag = builder.build()

    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=LIZ_CAPACITY,
        memory_capacity=MEMORY_CAPACITY,
        temp_capacity=TEMP_CAPACITY,
    )

    if initial_override is not None:
        initial = dict(initial_override)
    else:
        initial = {}
        for qid in circuit_module.DATA_QUBITS:
            initial[8 - qid] = 1
        for qid in circuit_module.ANCILLA_QUBITS:
            initial[qid] = 5

    config = CompilerConfig(
        interleave=interleave,
        measure_batch_size=batch_size,
        memory_reorder=memory_reorder,
    )
    compiler = QCCDCompiler(arch, config=config)
    result = compiler.compile(dag, initial)

    validation = validate_operations(result.operations, initial, arch, verbose=False)
    par_validation = validate_timesteps(result.timesteps, initial, arch, verbose=False)
    seq_est = Estimator().estimate(result.operations)
    par_est = ParallelEstimator().estimate(result.timesteps)

    # MS gate の実行順を記録
    ms_order = []
    for op in result.operations:
        if isinstance(op, GateOp) and op.type == 'MS':
            ms_order.append(op.targets)
        elif isinstance(op, MeasureOp):
            ms_order.append(('MEASURE', op.targets))

    return {
        "circuit": circuit_module_name,
        "config": config.to_dict(),
        "result": result,
        "initial": initial,
        "arch": arch,
        "validation": validation,
        "par_validation": par_validation,
        "seq_time": seq_est.total_time,
        "par_time": par_est.total_time,
        "par_estimation": par_est,
        "stats": result.stats,
        "total_ops": len(result.operations),
        "total_timesteps": len(result.timesteps),
        "ms_order": ms_order,
    }


def run_comparison(circuit_module_name: str):
    """1回路の比較実験"""

    print(f"\n{'=' * 70}")
    print(f"  {circuit_module_name}")
    print(f"{'=' * 70}")

    configs = [
        ("baseline", False, 1),
        ("interleave", True, 1),
        ("interleave+batch2", True, 2),
    ]

    results = {}
    for label, interleave, batch in configs:
        r = compile_circuit(circuit_module_name, interleave=interleave, batch_size=batch)
        results[label] = r
        status = "PASS" if (r["validation"].is_valid and r["par_validation"].is_valid) else "FAIL"
        print(f"  [{label}] Validation: {status}")
        if not r["validation"].is_valid or not r["par_validation"].is_valid:
            all_errors = r["validation"].errors + r["par_validation"].errors
            for e in all_errors[:3]:
                print(f"    {e}")

    # Summary table
    labels = list(results.keys())
    print()
    print(f"  {'Metric':<22}", end="")
    for label in labels:
        print(f" {label:>16}", end="")
    print()
    print(f"  {'-' * (22 + 16 * len(labels))}")

    metrics = [
        ("Total ops", "total_ops", False),
        ("Timesteps", "total_timesteps", False),
        ("Sequential (ms)", "seq_time", True),
        ("Parallel (ms)", "par_time", True),
    ]
    for name, key, is_float in metrics:
        print(f"  {name:<22}", end="")
        for label in labels:
            v = results[label][key]
            if is_float:
                print(f" {v / 1000:>16.2f}", end="")
            else:
                print(f" {v:>16}", end="")
        print()

    # Operation breakdown
    op_counts = {}
    for label in labels:
        op_counts[label] = Counter(type(op).__name__ for op in results[label]["result"].operations)
    all_types = sorted(set().union(*op_counts.values()))

    print()
    print(f"  {'Op Type':<15}", end="")
    for label in labels:
        print(f" {label:>16}", end="")
    print()
    print(f"  {'-' * (15 + 16 * len(labels))}")
    for t in all_types:
        print(f"  {t:<15}", end="")
        for label in labels:
            print(f" {op_counts[label][t]:>16}", end="")
        print()

    # MS gate interleave order
    print()
    print("  MS gate execution order (interleave):")
    for entry in results["interleave"]["ms_order"]:
        if isinstance(entry, tuple) and entry[0] == 'MEASURE':
            print(f"    MEASURE {entry[1]}")
        else:
            print(f"    MS {entry}")

    # Save outputs
    output_dir = Path(__file__).parent.parent / 'results' / 'interleave'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label in labels:
        r = results[label]
        txt_file = output_dir / f"{circuit_module_name}_interleave_{label}_{timestamp}.txt"
        save_parallel_txt(
            r["result"].timesteps, str(txt_file),
            initial_placement=r["initial"], arch=r["arch"],
        )

    return results


def run_all(circuit_names: list):
    """全回路の比較実験"""

    print("=" * 70)
    print("Stabilizer Interleave Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architecture: T({TEMP_CAPACITY})-M({MEMORY_CAPACITY})-T({TEMP_CAPACITY})"
          f"-L({LIZ_CAPACITY})-T({TEMP_CAPACITY})-M({MEMORY_CAPACITY})-T({TEMP_CAPACITY})")
    print("=" * 70)

    all_results = []
    for name in circuit_names:
        try:
            results = run_comparison(name)
            all_results.append((name, results))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save JSON summary
    output_dir = Path(__file__).parent.parent / 'results' / 'interleave'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment": "interleave_comparison",
        "generated_at": datetime.now().isoformat(),
        "architecture": {
            "type": "simple_mlm",
            "liz_capacity": LIZ_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
            "temp_capacity": TEMP_CAPACITY,
        },
        "results": [],
    }
    for name, results in all_results:
        entry = {"circuit": name}
        for label, r in results.items():
            entry[label] = {
                "config": r["config"],
                "total_ops": r["total_ops"],
                "total_timesteps": r["total_timesteps"],
                "sequential_time_ms": r["seq_time"] / 1000,
                "parallel_time_ms": r["par_time"] / 1000,
                "stats": r["stats"],
                "valid": r["validation"].is_valid and r["par_validation"].is_valid,
            }
        summary["results"].append(entry)

    json_file = output_dir / f"interleave_comparison_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Output] Summary saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Stabilizer Interleave Experiment')
    parser.add_argument('circuits', nargs='*', default=CIRCUIT_MODULES,
                        help='Circuit module names (default: all)')
    args = parser.parse_args()

    run_all(args.circuits)


if __name__ == "__main__":
    main()
