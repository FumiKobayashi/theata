"""
Surface-17 Syndrome Extraction Circuit (X/Z Interleaved Order)

スタビライザ順序: X0 → Z0 → X1 → Z1 → X2 → Z2 → X3 → Z3
  9 (X0X1X3X4) → 13 (Z0Z3) → 10 (X1X2) → 14 (Z1Z2Z4Z5) → 
  11 (X4X5X7X8) → 15 (Z3Z4Z6Z7) → 12 (X6X7) → 16 (Z5Z8)

この順序により、Data qubitのZone間移動が減少し、
Split/Merge操作が削減される。

Performance comparison (vs basic X→Z order):
  - Operations: 947 vs 1189 (-20.4%)
  - Time: 79.14 ms vs 99.50 ms (-20.5%)
  - Split/Merge: 59.40 ms vs 78.80 ms (-24.6%)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# srcディレクトリをパスに追加
# プロジェクトルート/experiments/surface17_basic.py から実行する想定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from estimator import Estimator


# Surface-17 回路定義 interleaved
SURFACE17_CIRCUIT = """
QUBIT 0 AT (0, 0)
QUBIT 1 AT (0, 1)
QUBIT 2 AT (0, 2)
QUBIT 3 AT (0, 3)
QUBIT 4 AT (0, 4)
QUBIT 5 AT (0, 5)
QUBIT 6 AT (0, 6)
QUBIT 7 AT (0, 7)
QUBIT 8 AT (0, 8)
QUBIT 9 AT (2, 0)
QUBIT 10 AT (2, 1)
QUBIT 11 AT (2, 2)
QUBIT 12 AT (2, 3)
QUBIT 13 AT (2, 4)
QUBIT 14 AT (2, 5)
QUBIT 15 AT (2, 6)
QUBIT 16 AT (2, 7)

RESET 9
R 9 1.5708 0.0
R 9 1.5708 1.5708
R 9 1.5708 1.5708
MS 9 1 0.25
R 9 1.5708 3.1416
R 1 1.5708 3.1416
R 9 1.5708 4.7124
R 9 1.5708 1.5708
MS 9 0 0.25
R 9 1.5708 3.1416
R 0 1.5708 3.1416
R 9 1.5708 4.7124
R 9 1.5708 1.5708
MS 9 4 0.25
R 9 1.5708 3.1416
R 4 1.5708 3.1416
R 9 1.5708 4.7124
R 9 1.5708 1.5708
MS 9 3 0.25
R 9 1.5708 3.1416
R 3 1.5708 3.1416
R 9 1.5708 4.7124
R 9 1.5708 0.0
R 9 1.5708 1.5708
MEASURE 9
RESET 13
R 13 1.5708 1.5708
MS 13 0 0.25
R 13 1.5708 3.1416
R 0 1.5708 3.1416
R 13 1.5708 4.7124
R 13 1.5708 1.5708
MS 13 3 0.25
R 13 1.5708 3.1416
R 3 1.5708 3.1416
R 13 1.5708 4.7124
MEASURE 13
RESET 10
R 10 1.5708 0.0
R 10 1.5708 1.5708
R 10 1.5708 1.5708
MS 10 1 0.25
R 10 1.5708 3.1416
R 1 1.5708 3.1416
R 10 1.5708 4.7124
R 10 1.5708 1.5708
MS 10 2 0.25
R 10 1.5708 3.1416
R 2 1.5708 3.1416
R 10 1.5708 4.7124
R 10 1.5708 0.0
R 10 1.5708 1.5708
MEASURE 10
RESET 14
R 14 1.5708 1.5708
MS 14 2 0.25
R 14 1.5708 3.1416
R 2 1.5708 3.1416
R 14 1.5708 4.7124
R 14 1.5708 1.5708
MS 14 1 0.25
R 14 1.5708 3.1416
R 1 1.5708 3.1416
R 14 1.5708 4.7124
R 14 1.5708 1.5708
MS 14 5 0.25
R 14 1.5708 3.1416
R 5 1.5708 3.1416
R 14 1.5708 4.7124
R 14 1.5708 1.5708
MS 14 4 0.25
R 14 1.5708 3.1416
R 4 1.5708 3.1416
R 14 1.5708 4.7124
MEASURE 14
RESET 11
R 11 1.5708 0.0
R 11 1.5708 1.5708
R 11 1.5708 1.5708
MS 11 5 0.25
R 11 1.5708 3.1416
R 5 1.5708 3.1416
R 11 1.5708 4.7124
R 11 1.5708 1.5708
MS 11 4 0.25
R 11 1.5708 3.1416
R 4 1.5708 3.1416
R 11 1.5708 4.7124
R 11 1.5708 1.5708
MS 11 8 0.25
R 11 1.5708 3.1416
R 8 1.5708 3.1416
R 11 1.5708 4.7124
R 11 1.5708 1.5708
MS 11 7 0.25
R 11 1.5708 3.1416
R 7 1.5708 3.1416
R 11 1.5708 4.7124
R 11 1.5708 0.0
R 11 1.5708 1.5708
MEASURE 11
RESET 15
R 15 1.5708 1.5708
MS 15 4 0.25
R 15 1.5708 3.1416
R 4 1.5708 3.1416
R 15 1.5708 4.7124
R 15 1.5708 1.5708
MS 15 3 0.25
R 15 1.5708 3.1416
R 3 1.5708 3.1416
R 15 1.5708 4.7124
R 15 1.5708 1.5708
MS 15 7 0.25
R 15 1.5708 3.1416
R 7 1.5708 3.1416
R 15 1.5708 4.7124
R 15 1.5708 1.5708
MS 15 6 0.25
R 15 1.5708 3.1416
R 6 1.5708 3.1416
R 15 1.5708 4.7124
MEASURE 15
RESET 12
R 12 1.5708 0.0
R 12 1.5708 1.5708
R 12 1.5708 1.5708
MS 12 6 0.25
R 12 1.5708 3.1416
R 6 1.5708 3.1416
R 12 1.5708 4.7124
R 12 1.5708 1.5708
MS 12 7 0.25
R 12 1.5708 3.1416
R 7 1.5708 3.1416
R 12 1.5708 4.7124
R 12 1.5708 0.0
R 12 1.5708 1.5708
MEASURE 12
RESET 16
R 16 1.5708 1.5708
MS 16 5 0.25
R 16 1.5708 3.1416
R 5 1.5708 3.1416
R 16 1.5708 4.7124
R 16 1.5708 1.5708
MS 16 8 0.25
R 16 1.5708 3.1416
R 8 1.5708 3.1416
R 16 1.5708 4.7124
MEASURE 16
"""


# ==========================================
# 出力機能
# ==========================================

def operation_to_dict(op) -> dict:
    """
    Operation を辞書に変換（JSON出力用）
    
    依存モジュールなしで動作するよう、型名の文字列判定を使用
    """
    type_name = type(op).__name__
    result = {"type": type_name}
    
    if type_name == "GateOp":
        result["gate_type"] = op.type  # "MS" or "R"
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
        # フォールバック: str表現を使用
        result["repr"] = str(op)
    
    return result


def save_operations_json(
    result,
    estimation,
    filepath: str,
    metadata: dict = None
):
    """
    コンパイル結果をJSONファイルに保存
    
    Args:
        result: CompileResult オブジェクト
        estimation: Estimator の結果
        filepath: 保存先パス
        metadata: 追加のメタデータ（オプション）
    """
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(result.operations),
            "stats": result.stats,
            "time_estimation": {
                "total_time_us": estimation.total_time,
                "total_time_ms": estimation.total_time / 1000,
                "breakdown": estimation.breakdown if hasattr(estimation, 'breakdown') else {}
            }
        },
        "operations": [operation_to_dict(op) for op in result.operations]
    }
    
    if metadata:
        output["metadata"].update(metadata)
    
    # ディレクトリがなければ作成
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Output] Saved to: {filepath}")


def run_experiment(
    liz_capacity: int = 4,
    memory_capacity: int = 30,
    placement_name: str = "default",
    config: CompilerConfig = None,
    output_dir: str = None
):
    """
    Surface-17のコンパイル実験を実行
    
    Args:
        liz_capacity: LIZの容量
        memory_capacity: Memoryゾーンの容量
        placement_name: 初期配置の名前（"default", "interleaved"など）
        config: コンパイラ設定（Noneならbaseline）
        output_dir: 結果を保存するディレクトリ（Noneなら保存しない）
    """
    config = config or CompilerConfig.baseline()
    
    print("=" * 70)
    print("Surface-17 Syndrome Extraction - Compilation Experiment")
    print(f"Config: {config}")
    print("=" * 70)
    
    # パース
    program = parse(SURFACE17_CIRCUIT)
    print(f"\n[Input Circuit]")
    print(f"  Qubits: {len(program.declarations)}")
    print(f"  Instructions: {len(program.instructions)}")
    
    # DAG構築
    builder = DAGBuilder(program)
    dag = builder.build()
    print(f"  DAG nodes: {len(dag.nodes)}")
    
    # アーキテクチャ
    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=liz_capacity,
        memory_capacity=memory_capacity
    )
    print(f"\n[Architecture]")
    print(f"  {arch}")
    print(f"  LIZ capacity: {liz_capacity}")
    print(f"  Memory capacity: {memory_capacity}")
    
    # 初期配置
    if placement_name == "default":
        # Data(0-8) → zone 0, Ancilla(9-16) → zone 2
        initial = {}
        for i in range(9):
            initial[i] = 0
        for i in range(9, 17):
            initial[i] = 2
    else:
        raise ValueError(f"Unknown placement: {placement_name}")
    
    print(f"\n[Initial Placement: {placement_name}]")
    print(f"  Data qubits (0-8): zone 0")
    print(f"  Ancilla qubits (9-16): zone 2")
    
    # コンパイル
    print(f"\n[Compiling...]")
    compiler = QCCDCompiler(arch, config=config)
    result = compiler.compile(dag, initial)
    
    print(f"\n[Compilation Result]")
    print(f"  Total operations: {len(result.operations)}")
    print(f"  Statistics:")
    for key, value in sorted(result.stats.items()):
        print(f"    {key}: {value}")
    
    # 時間推定
    estimator = Estimator()
    estimation = estimator.estimate(result.operations)
    
    print(f"\n[Time Estimation]")
    print(estimation)
    
    # サマリー
    print(f"\n[Summary]")
    print(f"  Input:  {len(program.instructions)} operations")
    print(f"  Output: {len(result.operations)} operations")
    print(f"  Expansion ratio: {len(result.operations) / len(program.instructions):.1f}x")
    print(f"  Total time: {estimation.total_time:.2f} μs = {estimation.total_time/1000:.2f} ms")
    
    # ファイル出力
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ファイル名にconfig情報を含める
        config_tag = f"gate-{config.gate_selection}"
        filename = f"surface17_liz{liz_capacity}_{config_tag}_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        save_operations_json(
            result=result,
            estimation=estimation,
            filepath=str(filepath),
            metadata={
                "experiment": "surface17_syndrome_extraction_interleaved",
                "config": config.to_dict(),  # ★ config を含める
                "liz_capacity": liz_capacity,
                "memory_capacity": memory_capacity,
                "placement": placement_name,
                "input_operations": len(program.instructions),
                "expansion_ratio": len(result.operations) / len(program.instructions)
            }
        )
    
    return result, estimation


def main():
    """メイン関数"""
    # 出力ディレクトリ（experiments/ と同階層の result/）
    output_dir = Path(__file__).parent.parent / "results"
    
    # 基本実験
    result, estimation = run_experiment(
        liz_capacity=4,
        memory_capacity=20,
        placement_name="default",
        output_dir=str(output_dir)
    )
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()