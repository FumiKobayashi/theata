"""
Surface-17 Syndrome Extraction Circuit - Basic Compilation Experiment

このスクリプトは、Surface-17のシンドローム抽出回路をQCCDアーキテクチャ上で
コンパイルし、時間推定を行う。

使用方法:
    python experiments/surface17_basic.py

    # Baseline
    config = CompilerConfig.baseline()

    # バッチ処理（LIZ容量でバッチ）
    config = CompilerConfig(batch_reset_measure=True)

    # バッチサイズ指定
    config = CompilerConfig(batch_reset_measure=True, batch_size=2)

出力:
    - コンパイル結果（操作数、統計情報）
    - 時間推定（内訳、合計）
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


# Surface-17 回路定義
SURFACE17_CIRCUIT = """
# ============================================================
# Surface-17 Syndrome Extraction Circuit (QCCD Native Format)
# ============================================================
# 
# Data qubit layout:
#     D0 --- D1 --- D2
#     |      |      |
#     D3 --- D4 --- D5
#     |      |      |
#     D6 --- D7 --- D8
#
# X Stabilizers (4):
#   A0 (9):  X0X1X3X4 (weight-4)
#   A1 (10): X1X2 (weight-2)
#   A2 (11): X4X5X7X8 (weight-4)
#   A3 (12): X6X7 (weight-2)
#
# Z Stabilizers (4):
#   A4 (13): Z0Z3 (weight-2)
#   A5 (14): Z1Z2Z4Z5 (weight-4)
#   A6 (15): Z3Z4Z6Z7 (weight-4)
#   A7 (16): Z5Z8 (weight-2)
#
# CNOT Implementation (Figure 1, Maslov 2017, v=1, s=1):
#   CNOT(c,t) = RY(π/2)_c · XX(π/4)_ct · RX(-π/2)_c · RX(-π/2)_t · RY(-π/2)_c
#   RY(θ) = R(θ, π/2), RX(-π/2) = R(π/2, π), RY(-π/2) = R(π/2, 3π/2)
# ============================================================

# --- Qubit Declarations (temporary placement) ---
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

# ============================================================
# X Stabilizer Measurements
# ============================================================

# --- X0X1X3X4 (Ancilla 9) ---
RESET 9
R 9 1.5708 0.0
R 9 1.5708 1.5708
# CNOT(9, 1)
R 9 1.5708 1.5708
MS 9 1 0.25
R 9 1.5708 3.1416
R 1 1.5708 3.1416
R 9 1.5708 4.7124
# CNOT(9, 0)
R 9 1.5708 1.5708
MS 9 0 0.25
R 9 1.5708 3.1416
R 0 1.5708 3.1416
R 9 1.5708 4.7124
# CNOT(9, 4)
R 9 1.5708 1.5708
MS 9 4 0.25
R 9 1.5708 3.1416
R 4 1.5708 3.1416
R 9 1.5708 4.7124
# CNOT(9, 3)
R 9 1.5708 1.5708
MS 9 3 0.25
R 9 1.5708 3.1416
R 3 1.5708 3.1416
R 9 1.5708 4.7124
R 9 1.5708 0.0
R 9 1.5708 1.5708
MEASURE 9

# --- X1X2 (Ancilla 10) ---
RESET 10
R 10 1.5708 0.0
R 10 1.5708 1.5708
# CNOT(10, 1)
R 10 1.5708 1.5708
MS 10 1 0.25
R 10 1.5708 3.1416
R 1 1.5708 3.1416
R 10 1.5708 4.7124
# CNOT(10, 2)
R 10 1.5708 1.5708
MS 10 2 0.25
R 10 1.5708 3.1416
R 2 1.5708 3.1416
R 10 1.5708 4.7124
R 10 1.5708 0.0
R 10 1.5708 1.5708
MEASURE 10

# --- X4X5X7X8 (Ancilla 11) ---
RESET 11
R 11 1.5708 0.0
R 11 1.5708 1.5708
# CNOT(11, 5)
R 11 1.5708 1.5708
MS 11 5 0.25
R 11 1.5708 3.1416
R 5 1.5708 3.1416
R 11 1.5708 4.7124
# CNOT(11, 4)
R 11 1.5708 1.5708
MS 11 4 0.25
R 11 1.5708 3.1416
R 4 1.5708 3.1416
R 11 1.5708 4.7124
# CNOT(11, 8)
R 11 1.5708 1.5708
MS 11 8 0.25
R 11 1.5708 3.1416
R 8 1.5708 3.1416
R 11 1.5708 4.7124
# CNOT(11, 7)
R 11 1.5708 1.5708
MS 11 7 0.25
R 11 1.5708 3.1416
R 7 1.5708 3.1416
R 11 1.5708 4.7124
R 11 1.5708 0.0
R 11 1.5708 1.5708
MEASURE 11

# --- X6X7 (Ancilla 12) ---
RESET 12
R 12 1.5708 0.0
R 12 1.5708 1.5708
# CNOT(12, 6)
R 12 1.5708 1.5708
MS 12 6 0.25
R 12 1.5708 3.1416
R 6 1.5708 3.1416
R 12 1.5708 4.7124
# CNOT(12, 7)
R 12 1.5708 1.5708
MS 12 7 0.25
R 12 1.5708 3.1416
R 7 1.5708 3.1416
R 12 1.5708 4.7124
R 12 1.5708 0.0
R 12 1.5708 1.5708
MEASURE 12

# ============================================================
# Z Stabilizer Measurements
# ============================================================

# --- Z0Z3 (Ancilla 13) ---
RESET 13
# CNOT(13, 0)
R 13 1.5708 1.5708
MS 13 0 0.25
R 13 1.5708 3.1416
R 0 1.5708 3.1416
R 13 1.5708 4.7124
# CNOT(13, 3)
R 13 1.5708 1.5708
MS 13 3 0.25
R 13 1.5708 3.1416
R 3 1.5708 3.1416
R 13 1.5708 4.7124
MEASURE 13

# --- Z1Z2Z4Z5 (Ancilla 14) ---
RESET 14
# CNOT(14, 2)
R 14 1.5708 1.5708
MS 14 2 0.25
R 14 1.5708 3.1416
R 2 1.5708 3.1416
R 14 1.5708 4.7124
# CNOT(14, 1)
R 14 1.5708 1.5708
MS 14 1 0.25
R 14 1.5708 3.1416
R 1 1.5708 3.1416
R 14 1.5708 4.7124
# CNOT(14, 5)
R 14 1.5708 1.5708
MS 14 5 0.25
R 14 1.5708 3.1416
R 5 1.5708 3.1416
R 14 1.5708 4.7124
# CNOT(14, 4)
R 14 1.5708 1.5708
MS 14 4 0.25
R 14 1.5708 3.1416
R 4 1.5708 3.1416
R 14 1.5708 4.7124
MEASURE 14

# --- Z3Z4Z6Z7 (Ancilla 15) ---
RESET 15
# CNOT(15, 4)
R 15 1.5708 1.5708
MS 15 4 0.25
R 15 1.5708 3.1416
R 4 1.5708 3.1416
R 15 1.5708 4.7124
# CNOT(15, 3)
R 15 1.5708 1.5708
MS 15 3 0.25
R 15 1.5708 3.1416
R 3 1.5708 3.1416
R 15 1.5708 4.7124
# CNOT(15, 7)
R 15 1.5708 1.5708
MS 15 7 0.25
R 15 1.5708 3.1416
R 7 1.5708 3.1416
R 15 1.5708 4.7124
# CNOT(15, 6)
R 15 1.5708 1.5708
MS 15 6 0.25
R 15 1.5708 3.1416
R 6 1.5708 3.1416
R 15 1.5708 4.7124
MEASURE 15

# --- Z5Z8 (Ancilla 16) ---
RESET 16
# CNOT(16, 5)
R 16 1.5708 1.5708
MS 16 5 0.25
R 16 1.5708 3.1416
R 5 1.5708 3.1416
R 16 1.5708 4.7124
# CNOT(16, 8)
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


# ==========================================
# DAG分析機能
# ==========================================

def analyze_dag(dag, strategy: str = "min_id") -> dict:
    """
    DAGの構造と実行シミュレーションを分析
    
    Args:
        dag: DAGオブジェクト
        strategy: ゲート選択戦略 ("min_id" のみ対応)
    
    Returns:
        分析結果の辞書
    """
    from ast_def import GateOp, ResetOp, MeasureOp
    import copy
    
    # DAGをコピーして分析（元のDAGを変更しない）
    dag_copy = copy.deepcopy(dag)
    dag_copy.reset_execution()
    
    # 基本統計
    num_nodes = len(dag_copy.nodes)
    num_edges = sum(len(n.successors) for n in dag_copy.nodes.values())
    
    # Front Layer推移をシミュレート
    front_layer_sizes = []
    front_layer_details = []  # 各イテレーションの詳細
    
    iteration = 0
    while not dag_copy.is_complete():
        front = dag_copy.get_front_layer()
        
        # 各タイプをカウント
        resets = [nid for nid in front if isinstance(dag_copy.nodes[nid].operation, ResetOp)]
        measures = [nid for nid in front if isinstance(dag_copy.nodes[nid].operation, MeasureOp)]
        gates = [nid for nid in front if isinstance(dag_copy.nodes[nid].operation, GateOp)]
        
        front_layer_sizes.append(len(front))
        front_layer_details.append({
            "iteration": iteration,
            "total": len(front),
            "resets": len(resets),
            "gates": len(gates),
            "measures": len(measures)
        })
        
        # 選択戦略に従って実行
        if strategy == "min_id":
            selected = min(front)
        else:
            selected = min(front)  # デフォルトはmin_id
        
        dag_copy.execute(selected)
        iteration += 1
    
    # 統計計算
    if front_layer_sizes:
        min_front = min(front_layer_sizes)
        max_front = max(front_layer_sizes)
        avg_front = sum(front_layer_sizes) / len(front_layer_sizes)
    else:
        min_front = max_front = avg_front = 0
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "total_iterations": iteration,
        "front_layer_stats": {
            "min": min_front,
            "max": max_front,
            "avg": round(avg_front, 2)
        },
        "front_layer_sizes": front_layer_sizes,
        "front_layer_details": front_layer_details
    }


def print_dag_analysis(analysis: dict, verbose: bool = False):
    """DAG分析結果を表示"""
    print(f"\n[DAG Analysis]")
    print(f"  Nodes: {analysis['num_nodes']}")
    print(f"  Edges: {analysis['num_edges']}")
    print(f"  Total iterations: {analysis['total_iterations']}")
    print(f"  Front Layer size: min={analysis['front_layer_stats']['min']}, "
          f"max={analysis['front_layer_stats']['max']}, "
          f"avg={analysis['front_layer_stats']['avg']}")
    
    if verbose:
        print(f"\n  Front Layer evolution (first 30 iterations):")
        for detail in analysis['front_layer_details'][:30]:
            print(f"    Iter {detail['iteration']:3d}: "
                  f"total={detail['total']:2d} "
                  f"(R:{detail['resets']}, G:{detail['gates']}, M:{detail['measures']})")
        if len(analysis['front_layer_details']) > 30:
            print(f"    ... ({len(analysis['front_layer_details']) - 30} more iterations)")


def run_experiment(
    liz_capacity: int = 4,
    memory_capacity: int = 30,
    placement_name: str = "default",
    config: CompilerConfig = None,
    output_dir: str = None,
    stabilizer_parallel: bool = False,
    analyze_dag_structure: bool = False,
    verbose_dag: bool = False
):
    """
    Surface-17のコンパイル実験を実行
    
    Args:
        liz_capacity: LIZの容量
        memory_capacity: Memoryゾーンの容量
        placement_name: 初期配置の名前（"default", "interleaved"など）
        config: コンパイラ設定（Noneならbaseline）
        output_dir: 結果を保存するディレクトリ（Noneなら保存しない）
        stabilizer_parallel: Trueの場合、スタビライザ単位で並列化したDAGを使用
        analyze_dag_structure: Trueの場合、DAG構造を分析して表示
        verbose_dag: Trueの場合、DAG分析の詳細（Front Layer推移）を表示
    """
    config = config or CompilerConfig.baseline()
    
    print("=" * 70)
    print("Surface-17 Syndrome Extraction - Compilation Experiment")
    print(f"Config: {config}")
    print(f"DAG mode: {'stabilizer_parallel' if stabilizer_parallel else 'normal'}")
    print("=" * 70)
    
    # パース
    program = parse(SURFACE17_CIRCUIT)
    print(f"\n[Input Circuit]")
    print(f"  Qubits: {len(program.declarations)}")
    print(f"  Instructions: {len(program.instructions)}")
    
    # DAG構築
    if stabilizer_parallel:
        builder = DAGBuilder(program, stabilizer_parallel=True)
    else:
        builder = DAGBuilder(program)
    dag = builder.build()
    print(f"  DAG nodes: {len(dag.nodes)}")
    print(f"  DAG mode: {'stabilizer_parallel' if stabilizer_parallel else 'normal'}")
    
    # DAG分析
    if analyze_dag_structure:
        dag_analysis = analyze_dag(dag)
        print_dag_analysis(dag_analysis, verbose=verbose_dag)
    
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
                "experiment": "surface17_syndrome_extraction",
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
    # 出力ディレクトリ（experiments/ と同階層の results/）
    output_dir = Path(__file__).parent.parent / "results"
    
    # ========================================
    # 実験設定（ここを変更して実験を切り替える）
    # ========================================
    # config = CompilerConfig.baseline()  # batch_reset_measure=False
    config = CompilerConfig(batch_reset_measure=True, batch_size=3)  # バッチ処理ON
    
    # ========================================
    
    print("\n" + "=" * 70)
    print(" Surface-17 Compilation Experiment")
    print("=" * 70)
    result, estimation = run_experiment(
        liz_capacity=4,
        memory_capacity=20,
        placement_name="default",
        config=config,
        stabilizer_parallel=True,
        analyze_dag_structure=True,
        verbose_dag=False,
        output_dir=output_dir  # JSON出力
    )
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()