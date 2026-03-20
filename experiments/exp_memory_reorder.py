"""
Memory Reorder Optimization Experiment

Measure/Reset中のMemory内先読み整列の効果を検証する。
Baseline (memory_reorder=False) vs Optimized (memory_reorder=True) を比較。

使い方:
    # 全回路を比較
    python exp_memory_reorder.py

    # 特定の回路のみ
    python exp_memory_reorder.py snake_per_stabilizer
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from estimator import Estimator
from parallel_estimator import ParallelEstimator, save_parallel_txt
from validator import validate_operations

# ==========================================
# 設定
# ==========================================

CIRCUIT_MODULES = [
    "snake_per_stabilizer",
    "snake_reuse1",
    "snake_reuse2",
    "snake_reuse4",
    "snake_batch_measure_per",
]

LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20
TEMP_CAPACITY = 20
STABILIZER_PARALLEL = False


# ==========================================
# 実験ロジック
# ==========================================

def compile_circuit(circuit_module_name: str, memory_reorder: bool):
    """回路をコンパイルして結果を返す"""
    circuit_module = importlib.import_module(circuit_module_name)

    program = parse(circuit_module.CIRCUIT)
    builder = DAGBuilder(program, stabilizer_parallel=STABILIZER_PARALLEL)
    dag = builder.build()

    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=LIZ_CAPACITY,
        memory_capacity=MEMORY_CAPACITY,
        temp_capacity=TEMP_CAPACITY,
    )

    initial = {}
    for qid in circuit_module.DATA_QUBITS:
        initial[8 - qid] = 1
    for qid in circuit_module.ANCILLA_QUBITS:
        initial[qid] = 5

    config = CompilerConfig(memory_reorder=memory_reorder)
    compiler = QCCDCompiler(arch, config=config)
    result = compiler.compile(dag, initial)

    # Validation
    validation = validate_operations(result.operations, initial, arch, verbose=False)

    # Time estimation
    seq_est = Estimator().estimate(result.operations)
    par_est = ParallelEstimator().estimate(result.timesteps)

    return {
        "circuit": circuit_module_name,
        "memory_reorder": memory_reorder,
        "result": result,
        "initial": initial,
        "arch": arch,
        "validation": validation,
        "seq_time": seq_est.total_time,
        "par_time": par_est.total_time,
        "par_estimation": par_est,
        "stats": result.stats,
        "total_ops": len(result.operations),
        "total_timesteps": len(result.timesteps),
    }


def count_ops_by_type(ops):
    """操作種別ごとのカウント"""
    from collections import Counter
    return Counter(type(op).__name__ for op in ops)


def run_comparison(circuit_module_name: str):
    """1回路の baseline vs optimized 比較"""

    print(f"\n{'=' * 70}")
    print(f"  {circuit_module_name}")
    print(f"{'=' * 70}")

    base = compile_circuit(circuit_module_name, memory_reorder=False)
    opt = compile_circuit(circuit_module_name, memory_reorder=True)

    # Validation check
    for r, label in [(base, "Baseline"), (opt, "Optimized")]:
        status = "PASS" if r["validation"].is_valid else "FAIL"
        print(f"  [{label}] Validation: {status}")
        if not r["validation"].is_valid:
            for e in r["validation"].errors[:3]:
                print(f"    {e}")

    # Summary table
    print()
    print(f"  {'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Diff':>10} {'Change':>8}")
    print(f"  {'-' * 67}")

    metrics = [
        ("Total ops", base["total_ops"], opt["total_ops"]),
        ("Timesteps", base["total_timesteps"], opt["total_timesteps"]),
        ("Sequential (ms)", base["seq_time"] / 1000, opt["seq_time"] / 1000),
        ("Parallel (ms)", base["par_time"] / 1000, opt["par_time"] / 1000),
    ]
    for name, b, o in metrics:
        diff = o - b
        pct = (diff / b * 100) if b != 0 else 0
        if isinstance(b, float):
            print(f"  {name:<25} {b:>12.2f} {o:>12.2f} {diff:>+10.2f} {pct:>+7.1f}%")
        else:
            print(f"  {name:<25} {b:>12} {o:>12} {diff:>+10} {pct:>+7.1f}%")

    # Operation breakdown
    base_c = count_ops_by_type(base["result"].operations)
    opt_c = count_ops_by_type(opt["result"].operations)
    all_types = sorted(set(base_c) | set(opt_c))

    print()
    print(f"  {'Op Type':<15} {'Baseline':>8} {'Optimized':>10} {'Diff':>8}")
    print(f"  {'-' * 41}")
    for t in all_types:
        b, o = base_c[t], opt_c[t]
        print(f"  {t:<15} {b:>8} {o:>10} {o - b:>+8}")

    # Parallelism distribution comparison
    from collections import Counter
    base_dist = Counter(d["ops_count"] for d in base["par_estimation"].timestep_details)
    opt_dist = Counter(d["ops_count"] for d in opt["par_estimation"].timestep_details)

    print()
    print(f"  Parallelism distribution:")
    print(f"  {'Ops/TS':<10} {'Baseline':>10} {'Optimized':>10}")
    all_keys = sorted(set(base_dist) | set(opt_dist))
    for k in all_keys:
        print(f"  {k:<10} {base_dist[k]:>10} {opt_dist[k]:>10}")

    # 結果を保存
    output_dir = Path(__file__).parent.parent / 'results' / 'memory_reorder'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r, label in [(base, "baseline"), (opt, "optimized")]:
        txt_file = output_dir / f"{circuit_module_name}_{label}_{timestamp}.txt"
        save_parallel_txt(
            r["result"].timesteps, str(txt_file),
            initial_placement=r["initial"], arch=r["arch"],
        )

    return base, opt


def run_all(circuit_names: list):
    """全回路の比較実験"""

    print("=" * 70)
    print("Memory Reorder Optimization Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architecture: T({TEMP_CAPACITY})-M({MEMORY_CAPACITY})-T({TEMP_CAPACITY})"
          f"-L({LIZ_CAPACITY})-T({TEMP_CAPACITY})-M({MEMORY_CAPACITY})-T({TEMP_CAPACITY})")
    print("=" * 70)

    results = []
    for name in circuit_names:
        try:
            base, opt = run_comparison(name)
            results.append((name, base, opt))
        except Exception as e:
            print(f"\n  ERROR: {e}")

    # Summary table across all circuits
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("  Overall Summary")
        print(f"{'=' * 70}")
        print()
        print(f"  {'Circuit':<30} {'Base(ms)':>10} {'Opt(ms)':>10} {'Saved(ms)':>10} {'Change':>8}")
        print(f"  {'-' * 68}")

        for name, base, opt in results:
            b = base["par_time"] / 1000
            o = opt["par_time"] / 1000
            diff = o - b
            pct = (diff / b * 100) if b != 0 else 0
            print(f"  {name:<30} {b:>10.2f} {o:>10.2f} {diff:>+10.2f} {pct:>+7.1f}%")

    # Save JSON summary
    output_dir = Path(__file__).parent.parent / 'results' / 'memory_reorder'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment": "memory_reorder_comparison",
        "generated_at": datetime.now().isoformat(),
        "architecture": {
            "type": "simple_mlm",
            "liz_capacity": LIZ_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
            "temp_capacity": TEMP_CAPACITY,
        },
        "results": [],
    }
    for name, base, opt in results:
        summary["results"].append({
            "circuit": name,
            "baseline": {
                "total_ops": base["total_ops"],
                "total_timesteps": base["total_timesteps"],
                "sequential_time_ms": base["seq_time"] / 1000,
                "parallel_time_ms": base["par_time"] / 1000,
                "stats": base["stats"],
                "valid": base["validation"].is_valid,
            },
            "optimized": {
                "total_ops": opt["total_ops"],
                "total_timesteps": opt["total_timesteps"],
                "sequential_time_ms": opt["seq_time"] / 1000,
                "parallel_time_ms": opt["par_time"] / 1000,
                "stats": opt["stats"],
                "valid": opt["validation"].is_valid,
            },
        })

    json_file = output_dir / f"memory_reorder_comparison_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Output] Summary saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Memory Reorder Optimization Experiment')
    parser.add_argument('circuits', nargs='*', default=CIRCUIT_MODULES,
                        help='Circuit module names (default: all)')
    args = parser.parse_args()

    run_all(args.circuits)


if __name__ == "__main__":
    main()
