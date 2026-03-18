"""
Parallel Analysis Experiment

単一の回路に対して詳細分析を行い、並列実行を考慮した結果を出力する。
TimeStep構造を保持したJSON/テキストを生成し、逐次実行との比較も行う。

使い方:
    # スクリプト内のCIRCUIT_MODULEを変更して実行
    python exp_parallel.py

    # または引数で指定
    python exp_parallel.py snake_per_stabilizer
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
from parallel_estimator import ParallelEstimator, format_timesteps_text, save_parallel_txt
from validator import validate_operations


# ==========================================
# 設定（ここを変更して実験）
# ==========================================

# 回路モジュール名（circuits/以下のファイル名、拡張子なし）
CIRCUIT_MODULE = "snake_batch_measure_per"

# コンパイラ設定
CONFIG = CompilerConfig(
    gate_selection="min_id",
)

# アーキテクチャ設定
LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20
TEMP_CAPACITY = 20

# DAG設定
STABILIZER_PARALLEL = False


# ==========================================
# 出力機能
# ==========================================

def operation_to_dict(op, arch=None) -> dict:
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
        result["dst_zone_id"] = op.dst_zone_id
    elif type_name == "MergeOp":
        result["zone_left"] = op.zone_left
        result["zone_right"] = op.zone_right
        if arch is not None:
            if arch.is_temp(op.zone_left) and not arch.is_temp(op.zone_right):
                result["dst_zone_id"] = op.zone_right
            elif arch.is_temp(op.zone_right) and not arch.is_temp(op.zone_left):
                result["dst_zone_id"] = op.zone_left
    elif type_name == "ResetOp":
        result["targets"] = op.targets
    elif type_name == "MeasureOp":
        result["targets"] = op.targets
    else:
        result["repr"] = str(op)

    return result


def save_parallel_json(result, seq_estimation, par_estimation, filepath: str,
                       arch=None, metadata: dict = None):
    """コンパイル結果をJSON保存（TimeStep構造付き）"""
    # TimeStep単位の構造
    timesteps_data = []
    for ts_idx, ts in enumerate(result.timesteps):
        ts_data = {
            "index": ts_idx,
            "ops_count": len(ts.ops),
            "ops": [operation_to_dict(op, arch=arch) for op in ts.ops],
        }
        timesteps_data.append(ts_data)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_operations": par_estimation.total_operations,
            "total_timesteps": par_estimation.total_timesteps,
            "stats": result.stats,
            "time_estimation": {
                "sequential": {
                    "total_time_us": seq_estimation.total_time,
                    "total_time_ms": seq_estimation.total_time / 1000,
                    "counts": seq_estimation.counts,
                    "breakdown": seq_estimation.breakdown,
                },
                "parallel": {
                    "total_time_us": par_estimation.total_time,
                    "total_time_ms": par_estimation.total_time / 1000,
                    "speedup": par_estimation.speedup,
                    "avg_parallelism": par_estimation.parallelism,
                    "counts": par_estimation.counts,
                    "breakdown": par_estimation.breakdown,
                },
            },
        },
        "timesteps": timesteps_data,
    }

    if metadata:
        output["metadata"].update(metadata)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Output] Saved to: {filepath}")


# ==========================================
# メイン
# ==========================================

def run_parallel_analysis(circuit_module_name: str):
    """並列分析を実行"""

    print("=" * 70)
    print("Parallel Analysis Experiment")
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
        memory_capacity=MEMORY_CAPACITY,
        temp_capacity=TEMP_CAPACITY,
    )
    print(f"\n[Architecture]")
    print(f"  {arch}")
    print(f"  LIZ capacity: {LIZ_CAPACITY}")
    print(f"  Memory capacity: {MEMORY_CAPACITY}")
    print(f"  Temp capacity: {TEMP_CAPACITY}")

    # 初期配置（Memory Zone: zone 1 = 左Memory, zone 5 = 右Memory）
    initial = {}
    for qid in circuit_module.DATA_QUBITS:
        initial[8-qid] = 1
    for qid in circuit_module.ANCILLA_QUBITS:
        initial[qid] = 5

    print(f"\n[Initial Placement]")
    print(f"  Data qubits -> zone 1 (Memory left)")
    print(f"  Ancilla qubits -> zone 5 (Memory right)")

    # コンパイラ設定
    print(f"\n[Compiler Config]")
    print(f"  {CONFIG}")

    # コンパイル
    print(f"\n[Compiling...]")
    compiler = QCCDCompiler(arch, config=CONFIG)
    result = compiler.compile(dag, initial)

    print(f"\n[Compilation Result]")
    print(f"  Total operations: {len(result.operations)}")
    print(f"  Total timesteps: {len(result.timesteps)}")
    print(f"  Statistics:")
    for key, value in sorted(result.stats.items()):
        print(f"    {key}: {value}")

    # 検証
    print(f"\n[Validation]")
    validation = validate_operations(result.operations, initial, arch, verbose=False)
    if validation.is_valid:
        print(f"  PASSED ({len(result.operations)} operations verified)")
    else:
        print(f"  FAILED: {len(validation.errors)} error(s)")
        for e in validation.errors[:10]:
            print(f"    {e}")
        if len(validation.errors) > 10:
            print(f"    ... and {len(validation.errors) - 10} more")

    # 時間推定（逐次 vs 並列）
    seq_estimation = Estimator().estimate(result.operations)
    par_estimator = ParallelEstimator()
    par_estimation = par_estimator.estimate(result.timesteps)

    print(f"\n[Time Estimation: Sequential vs Parallel]")
    print(f"  Sequential: {seq_estimation.total_time:.2f} us = {seq_estimation.total_time/1000:.2f} ms")
    print(f"  Parallel:   {par_estimation.total_time:.2f} us = {par_estimation.total_time/1000:.2f} ms")
    print(f"  Speedup:    {par_estimation.speedup:.2f}x")
    print(f"  Avg parallelism: {par_estimation.parallelism:.2f} ops/timestep")

    # 並列度の分布
    from collections import Counter
    par_dist = Counter(detail["ops_count"] for detail in par_estimation.timestep_details)
    print(f"\n[Parallelism Distribution]")
    for k in sorted(par_dist):
        count = par_dist[k]
        pct = count / par_estimation.total_timesteps * 100
        bar = "#" * int(pct / 2)
        print(f"  {k} ops: {count:>4} timesteps ({pct:5.1f}%) {bar}")

    # Breakdown比較
    print(f"\n  {'Operation':<12} {'Count':>6} {'Seq(us)':>12} {'Par(us)':>12} {'Saved(us)':>12}")
    print(f"  {'-'*56}")

    # 並列推定のTimestep単位のop別時間を計算
    par_op_time = {}
    for detail in par_estimation.timestep_details:
        ops = detail["ops"]
        if len(ops) == 1:
            # 単独実行: そのopの時間がそのまま
            op = ops[0]
            par_op_time[op["type"]] = par_op_time.get(op["type"], 0.0) + op["time"]
        else:
            # 並列実行: 最大時間のopがフル、それ以外は0（並列で隠れる）
            max_time = max(op["time"] for op in ops)
            max_op = max(ops, key=lambda o: o["time"])
            par_op_time[max_op["type"]] = par_op_time.get(max_op["type"], 0.0) + max_time
            for op in ops:
                if op is not max_op:
                    par_op_time[op["type"]] = par_op_time.get(op["type"], 0.0)

    for op_type in sorted(seq_estimation.breakdown.keys(), key=lambda x: -seq_estimation.breakdown[x]):
        count = seq_estimation.counts.get(op_type, 0)
        seq_t = seq_estimation.breakdown.get(op_type, 0.0)
        par_t = par_op_time.get(op_type, 0.0)
        saved = seq_t - par_t
        print(f"  {op_type:<12} {count:>6} {seq_t:>12.2f} {par_t:>12.2f} {saved:>12.2f}")

    # サマリー
    print(f"\n[Summary]")
    print(f"  Input:  {len(program.instructions)} instructions")
    print(f"  Output: {len(result.operations)} operations / {len(result.timesteps)} timesteps")
    print(f"  Expansion ratio: {len(result.operations) / len(program.instructions):.1f}x")
    print(f"  Sequential time: {seq_estimation.total_time:.2f} us = {seq_estimation.total_time/1000:.2f} ms")
    print(f"  Parallel time:   {par_estimation.total_time:.2f} us = {par_estimation.total_time/1000:.2f} ms")
    print(f"  Speedup:         {par_estimation.speedup:.2f}x")

    # 出力
    output_dir = Path(__file__).parent.parent / 'results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            "temp_capacity": TEMP_CAPACITY,
        },
        "stabilizer_parallel": STABILIZER_PARALLEL,
    }

    # JSON出力（TimeStep構造付き）
    json_file = output_dir / f"{circuit_module_name}_parallel_{timestamp}.json"
    save_parallel_json(result, seq_estimation, par_estimation, str(json_file),
                       arch=arch, metadata=metadata)

    # テキスト出力（TimeStepビュー）
    txt_file = output_dir / f"{circuit_module_name}_parallel_{timestamp}.txt"
    save_parallel_txt(result.timesteps, str(txt_file),
                      initial_placement=initial, arch=arch, estimator=par_estimator)

    print("\n" + "=" * 70)
    print("Parallel Analysis Complete")
    print("=" * 70)

    return result, seq_estimation, par_estimation


def main():
    parser = argparse.ArgumentParser(description='Parallel circuit analysis')
    parser.add_argument('circuit', nargs='?', default=CIRCUIT_MODULE,
                        help='Circuit module name (default: %(default)s)')
    args = parser.parse_args()

    run_parallel_analysis(args.circuit)


if __name__ == "__main__":
    main()
