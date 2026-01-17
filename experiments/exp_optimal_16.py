"""
16パターン一括実験スクリプト

最適化回路16パターンを一気に実験し、全てのJSONファイルを出力する。
最後にサマリーテーブルも表示。

使い方:
    python exp_optimal_16.py
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import importlib

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from estimator import Estimator


# ==========================================
# 設定
# ==========================================

# 16パターンの回路モジュール名
PATTERN_NAMES = [
    f"optimal_X{x}_Z{z}" 
    for x in ["A", "B", "C", "D"] 
    for z in ["A", "B", "C", "D"]
]

# コンパイラ設定
CONFIG = CompilerConfig(
    gate_selection="min_id",
    eviction_policy="first",
    target_liz_selection="first_available",
    batch_reset_measure=False,
    batch_size=0,
)

# アーキテクチャ設定
LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20

# DAG設定
STABILIZER_PARALLEL = True


# ==========================================
# Swap パターン分析
# ==========================================

def analyze_swap_patterns(operations) -> dict:
    """Swap操作の前後にあるSplit/Mergeのパターンを分析"""
    pattern_counts = {}
    total_swaps = 0
    
    op_types = [type(op).__name__ for op in operations]
    
    for i, op_type in enumerate(op_types):
        if op_type == "SwapOp":
            total_swaps += 1
            
            splits_before = 0
            j = i - 1
            while j >= 0 and op_types[j] == "SplitOp":
                splits_before += 1
                j -= 1
            
            merges_after = 0
            j = i + 1
            while j < len(op_types) and op_types[j] == "MergeOp":
                merges_after += 1
                j += 1
            
            pattern = f"{splits_before}-{merges_after}"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    result = {
        "total_swaps": total_swaps,
        "patterns": {},
    }
    
    for pattern, count in sorted(pattern_counts.items()):
        ratio = count / total_swaps * 100 if total_swaps > 0 else 0
        result["patterns"][pattern] = {
            "count": count,
            "ratio_percent": round(ratio, 2),
        }
    
    return result


# ==========================================
# 出力機能
# ==========================================

def operation_to_dict(op) -> dict:
    """Operation を辞書に変換"""
    type_name = type(op).__name__
    result = {"type": type_name}
    
    if type_name == "GateOp":
        result["gate_type"] = op.type
        result["targets"] = op.targets
        result["params"] = op.params
    elif type_name == "ShuttleOp":
        result["target"] = op.target
        result["dst"] = {"trap": op.dst.trap, "site": op.dst.site}
    elif type_name == "SwapOp":
        result["targets"] = op.targets
    elif type_name == "SplitOp":
        result["ion_id"] = op.ion_id
        result["right_zone_id"] = op.right_zone_id
    elif type_name == "MergeOp":
        result["zone_left"] = op.zone_left
        result["zone_right"] = op.zone_right
    elif type_name == "ResetOp":
        result["targets"] = op.targets
    elif type_name == "MeasureOp":
        result["targets"] = op.targets
    else:
        result["repr"] = str(op)
    
    return result


def save_detailed_json(result, estimation, filepath: str, metadata: dict = None, swap_analysis: dict = None):
    """コンパイル結果をJSONファイルに保存"""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(result.operations),
            "stats": result.stats,
            "time_estimation": {
                "total_time_us": estimation.total_time,
                "total_time_ms": estimation.total_time / 1000,
                "counts": estimation.counts,
                "breakdown": estimation.breakdown,
            }
        },
        "operations": [operation_to_dict(op) for op in result.operations]
    }
    
    if swap_analysis:
        output["metadata"]["swap_pattern_analysis"] = swap_analysis
    
    if metadata:
        output["metadata"].update(metadata)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ==========================================
# 単一パターンの実験
# ==========================================

def run_single_experiment(circuit_module_name: str, output_dir: Path, verbose: bool = False):
    """単一パターンの実験を実行"""
    
    # 回路モジュールをインポート
    circuit_module = importlib.import_module(circuit_module_name)
    
    # パース
    program = parse(circuit_module.CIRCUIT)
    
    # DAG構築
    builder = DAGBuilder(program, stabilizer_parallel=STABILIZER_PARALLEL)
    dag = builder.build()
    
    # アーキテクチャ
    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=LIZ_CAPACITY,
        memory_capacity=MEMORY_CAPACITY
    )
    
    # 初期配置
    initial = {}
    for qid in circuit_module.DATA_QUBITS:
        initial[qid] = 0
    for qid in circuit_module.ANCILLA_QUBITS:
        initial[qid] = 2
    
    # コンパイル
    compiler = QCCDCompiler(arch, config=CONFIG)
    result = compiler.compile(dag, initial)
    
    # 時間推定
    estimation = Estimator().estimate(result.operations)
    
    # Swapパターン分析
    swap_analysis = analyze_swap_patterns(result.operations)
    
    # メタデータ
    metadata = {
        "circuit_module": circuit_module_name,
        "qubit_count": circuit_module.QUBIT_COUNT,
        "data_qubits": circuit_module.DATA_QUBITS,
        "ancilla_qubits": circuit_module.ANCILLA_QUBITS,
        "config": CONFIG.to_dict(),
        "architecture": {
            "type": "simple_mlm",
            "liz_capacity": LIZ_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
        },
        "stabilizer_parallel": STABILIZER_PARALLEL,
    }
    
    # JSON出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{circuit_module_name}_{timestamp}.json"
    save_detailed_json(result, estimation, str(output_file), metadata, swap_analysis)
    
    # 結果サマリー
    summary = {
        "name": circuit_module_name,
        "total_ops": len(result.operations),
        "total_time_us": estimation.total_time,
        "swaps": result.stats.get("swaps", 0),
        "splits": result.stats.get("splits", 0),
        "merges": result.stats.get("merges", 0),
        "swap_patterns": swap_analysis["patterns"],
        "output_file": str(output_file),
    }
    
    if verbose:
        print(f"  {circuit_module_name}: {estimation.total_time:.2f} μs, {len(result.operations)} ops")
    
    return summary


# ==========================================
# メイン
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Run 16 optimal pattern experiments')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Optimal 16 Patterns Batch Experiment")
    print("=" * 70)
    print()
    print(f"Compiler Config: {CONFIG}")
    print(f"LIZ Capacity: {LIZ_CAPACITY}")
    print(f"Memory Capacity: {MEMORY_CAPACITY}")
    print(f"Stabilizer Parallel: {STABILIZER_PARALLEL}")
    print()
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("[Running experiments...]")
    for name in PATTERN_NAMES:
        try:
            summary = run_single_experiment(name, output_dir, verbose=args.verbose)
            results.append(summary)
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    
    # 時間でソート
    results_sorted = sorted(results, key=lambda x: x["total_time_us"])
    
    # テーブル表示
    print(f"{'Rank':<5} {'Pattern':<20} {'Time(μs)':>12} {'Time(ms)':>10} {'Ops':>8} {'Swaps':>8} {'Splits':>8}")
    print("-" * 75)
    
    for i, r in enumerate(results_sorted):
        print(f"{i+1:<5} {r['name']:<20} {r['total_time_us']:>12.2f} {r['total_time_us']/1000:>10.2f} {r['total_ops']:>8} {r['swaps']:>8} {r['splits']:>8}")
    
    print()
    
    # 最速と最遅
    fastest = results_sorted[0]
    slowest = results_sorted[-1]
    
    print(f"Fastest: {fastest['name']} ({fastest['total_time_us']:.2f} μs)")
    print(f"Slowest: {slowest['name']} ({slowest['total_time_us']:.2f} μs)")
    print(f"Difference: {slowest['total_time_us'] - fastest['total_time_us']:.2f} μs ({(slowest['total_time_us']/fastest['total_time_us'] - 1)*100:.1f}%)")
    
    print()
    
    # Swap パターン詳細（最速のみ）
    print(f"Fastest ({fastest['name']}) Swap Patterns:")
    for pattern, data in sorted(fastest['swap_patterns'].items()):
        print(f"  {pattern}: {data['count']} ({data['ratio_percent']:.1f}%)")
    
    print()
    print("=" * 70)
    print(f"All results saved to: {output_dir}")
    print("=" * 70)
    
    # サマリーJSONも保存
    summary_file = output_dir / f"optimal_16_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "compiler": CONFIG.to_dict(),
                "liz_capacity": LIZ_CAPACITY,
                "memory_capacity": MEMORY_CAPACITY,
                "stabilizer_parallel": STABILIZER_PARALLEL,
            },
            "results": results_sorted,
        }, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
