"""
Detailed Analysis Experiment

単一の回路に対して詳細分析を行い、命令列を含むJSONを出力する。

使い方:
    # スクリプト内のCIRCUIT_MODULEを変更して実行
    python exp_detailed.py

    # または引数で指定
    python exp_detailed.py surface_d3_reuse4_interleaved
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
# 設定（ここを変更して実験）
# ==========================================

# 回路モジュール名（circuits/以下のファイル名、拡張子なし）
CIRCUIT_MODULE = "snake_batch_measure"

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
# 出力機能
# ==========================================

def operation_to_dict(op) -> dict:
    """Operation を辞書に変換（JSON出力用）"""
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


def analyze_swap_patterns(operations) -> dict:
    """
    Swap操作の前後にあるSplit/Mergeのパターンを分析する。
    
    パターン:
      - "2-2": Split -> Split -> Swap -> Merge -> Merge
      - "1-1": Split -> Swap -> Merge
      - "0-0": Swap のみ（前後にSplit/Mergeなし）
      - その他: "n-m" 形式で記録
    
    Returns:
        dict: パターンごとのカウントと割合
    """
    pattern_counts = {}
    total_swaps = 0
    
    # 操作タイプ名のリストを作成
    op_types = [type(op).__name__ for op in operations]
    
    # 各Swapを検出してパターンを分析
    for i, op_type in enumerate(op_types):
        if op_type == "SwapOp":
            total_swaps += 1
            
            # 前方のSplit数をカウント（連続するSplitのみ）
            splits_before = 0
            j = i - 1
            while j >= 0 and op_types[j] == "SplitOp":
                splits_before += 1
                j -= 1
            
            # 後方のMerge数をカウント（連続するMergeのみ）
            merges_after = 0
            j = i + 1
            while j < len(op_types) and op_types[j] == "MergeOp":
                merges_after += 1
                j += 1
            
            # パターン名を作成
            pattern = f"{splits_before}-{merges_after}"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # 結果を整理
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


def save_detailed_json(result, estimation, filepath: str, metadata: dict = None, swap_analysis: dict = None):
    """コンパイル結果をJSONファイルに保存（命令列込み）"""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(result.operations),
            "stats": result.stats,
            "time_estimation": {
                "total_time_us": estimation.total_time,
                "total_time_ms": estimation.total_time / 1000,
                "counts": estimation.counts,
                "breakdown": estimation.breakdown,  # 操作種別ごとの時間（μs）
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
    
    print(f"\n[Output] Saved to: {filepath}")


# ==========================================
# メイン
# ==========================================

def run_detailed_analysis(circuit_module_name: str):
    """詳細分析を実行"""
    
    print("=" * 70)
    print("Detailed Analysis Experiment")
    print("=" * 70)
    
    # 回路モジュールをインポート
    circuit_module = importlib.import_module(circuit_module_name)
    
    print(f"\n[Circuit Module]")
    print(f"  Module: {circuit_module_name}")
    print(f"  Qubits: {circuit_module.QUBIT_COUNT}")
    print(f"  Data qubits: {circuit_module.DATA_QUBITS}")
    print(f"  Ancilla qubits: {circuit_module.ANCILLA_QUBITS}")
    
    # パース
    program = parse(circuit_module.CIRCUIT)
    print(f"\n[Input Circuit]")
    print(f"  Instructions: {len(program.instructions)}")
    
    # DAG構築
    builder = DAGBuilder(program, stabilizer_parallel=STABILIZER_PARALLEL)
    dag = builder.build()
    print(f"  DAG nodes: {len(dag.nodes)}")
    print(f"  DAG mode: {'stabilizer_parallel' if STABILIZER_PARALLEL else 'normal'}")
    
    # アーキテクチャ
    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=LIZ_CAPACITY,
        memory_capacity=MEMORY_CAPACITY
    )
    print(f"\n[Architecture]")
    print(f"  {arch}")
    print(f"  LIZ capacity: {LIZ_CAPACITY}")
    print(f"  Memory capacity: {MEMORY_CAPACITY}")
    
    # 初期配置
    initial = {}
    for qid in circuit_module.DATA_QUBITS:
        initial[8-qid] = 0
    for qid in circuit_module.ANCILLA_QUBITS:
        initial[qid] = 2
    
    print(f"\n[Initial Placement]")
    print(f"  Data qubits → zone 0")
    print(f"  Ancilla qubits → zone 2")
    
    # コンパイラ設定
    print(f"\n[Compiler Config]")
    print(f"  {CONFIG}")
    
    # コンパイル
    print(f"\n[Compiling...]")
    compiler = QCCDCompiler(arch, config=CONFIG)
    result = compiler.compile(dag, initial)
    
    print(f"\n[Compilation Result]")
    print(f"  Total operations: {len(result.operations)}")
    print(f"  Statistics:")
    for key, value in sorted(result.stats.items()):
        print(f"    {key}: {value}")
    
    # 時間推定
    estimation = Estimator().estimate(result.operations)
    
    print(f"\n[Time Estimation]")
    print(f"  Total: {estimation.total_time:.2f} μs = {estimation.total_time/1000:.2f} ms")
    
    # 詳細なbreakdownを表示
    total = estimation.total_time
    print(f"\n  {'Operation':<12} {'Count':>8} {'Time(μs)':>12} {'Ratio':>8}")
    print(f"  {'-'*44}")
    
    for op_type, time in sorted(estimation.breakdown.items(), key=lambda x: -x[1]):
        count = estimation.counts.get(op_type, 0)
        ratio = time / total * 100 if total > 0 else 0
        print(f"  {op_type:<12} {count:>8} {time:>12.2f} {ratio:>7.1f}%")
    
    # Swapパターン分析
    swap_analysis = analyze_swap_patterns(result.operations)
    
    print(f"\n[Swap Pattern Analysis]")
    print(f"  Total swaps: {swap_analysis['total_swaps']}")
    print(f"  Patterns (splits_before - merges_after):")
    for pattern, data in sorted(swap_analysis['patterns'].items()):
        print(f"    {pattern}: {data['count']} ({data['ratio_percent']:.1f}%)")
    
    # サマリー
    print(f"\n[Summary]")
    print(f"  Input:  {len(program.instructions)} operations")
    print(f"  Output: {len(result.operations)} operations")
    print(f"  Expansion ratio: {len(result.operations) / len(program.instructions):.1f}x")
    print(f"  Total time: {estimation.total_time:.2f} μs = {estimation.total_time/1000:.2f} ms")
    
    # JSON出力
    output_dir = Path(__file__).parent.parent / 'results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{circuit_module_name}_{timestamp}.json"
    
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
    
    save_detailed_json(result, estimation, str(output_file), metadata, swap_analysis)
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    
    return result, estimation


def main():
    parser = argparse.ArgumentParser(description='Detailed circuit analysis')
    parser.add_argument('circuit', nargs='?', default=CIRCUIT_MODULE,
                        help='Circuit module name (default: %(default)s)')
    args = parser.parse_args()
    
    run_detailed_analysis(args.circuit)


if __name__ == "__main__":
    main()