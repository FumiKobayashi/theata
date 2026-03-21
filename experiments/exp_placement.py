"""
Initial Placement Experiment

初期配置がコンパイル結果に与える影響を検証する。
以下の配置戦略を比較:
  1. baseline: データ=zone1(左Memory), アンシラ=zone5(右Memory)
  2. split_even: データとアンシラを左右均等に分散
  3. by_stabilizer: スタビライザペアごとに左右に振り分け
  4. random: ランダム配置 N 回のサンプリング

各配置戦略に対して、sequential / interleave の両方で比較可能。

使い方:
    # 全配置戦略を比較（sequential）
    python exp_placement.py

    # interleave有効で比較
    python exp_placement.py --interleave

    # ランダム配置のサンプル数を変更
    python exp_placement.py --random-samples 50

    # 特定の回路のみ
    python exp_placement.py --circuit snake_per_stabilizer
"""

import sys
import argparse
import random
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

# ==========================================
# 設定
# ==========================================

LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20
TEMP_CAPACITY = 20
RANDOM_SAMPLES = 20
RANDOM_SEED = 42

# Memory zones: zone 1 (left), zone 5 (right)
MEMORY_LEFT = 1
MEMORY_RIGHT = 5


# ==========================================
# 配置生成
# ==========================================

def placement_baseline(data_qubits, ancilla_qubits):
    """従来配置: データ=左Memory（容量超過時は右に溢れ）, アンシラ=右Memory

    データは逆順（大きいIDが左端）で左Memoryに配置。
    容量を超える分は右Memoryに溢れる。
    """
    initial = {}
    # データを逆順で左Memoryに配置（容量制限あり）
    sorted_data = sorted(data_qubits, reverse=True)
    for i, qid in enumerate(sorted_data):
        if i < MEMORY_CAPACITY:
            initial[qid] = MEMORY_LEFT
        else:
            initial[qid] = MEMORY_RIGHT
    # アンシラを右Memoryに配置
    for qid in ancilla_qubits:
        initial[qid] = MEMORY_RIGHT
    return _enforce_capacity(initial)


def placement_split_even(data_qubits, ancilla_qubits):
    """均等分散: データを左右に半分ずつ、アンシラはLIZに近い端に配置

    左Memory: [データ前半..., アンシラ前半] (アンシラがLIZ側=右端)
    右Memory: [アンシラ後半, データ後半...] (アンシラがLIZ側=左端)
    """
    initial = {}
    n_data = len(data_qubits)
    n_anc = len(ancilla_qubits)
    data_half = n_data // 2
    anc_half = (n_anc + 1) // 2  # 奇数の場合は左に多く

    # 左Memory: データ前半 → アンシラ前半（アンシラがLIZ側の右端に来る）
    for qid in data_qubits[:data_half]:
        initial[qid] = MEMORY_LEFT
    for qid in ancilla_qubits[:anc_half]:
        initial[qid] = MEMORY_LEFT

    # 右Memory: アンシラ後半 → データ後半（アンシラがLIZ側の左端に来る）
    for qid in ancilla_qubits[anc_half:]:
        initial[qid] = MEMORY_RIGHT
    for qid in data_qubits[data_half:]:
        initial[qid] = MEMORY_RIGHT

    return _enforce_capacity(initial)


def placement_by_stabilizer(data_qubits, ancilla_qubits, circuit_module):
    """
    スタビライザペアベース:
    奇数番目のスタビライザのイオン → 左Memory
    偶数番目のスタビライザのイオン → 右Memory
    共有qubitは先に登場した側に配置
    """
    program = parse(circuit_module.CIRCUIT)
    builder = DAGBuilder(program, stabilizer_parallel=True)
    builder.build()

    pairs = builder.get_stabilizer_pairs()
    initial = {}

    # スタビライザごとにMS gateからデータqubitを抽出
    stab_qubits = {}
    for node in builder.dag.nodes.values():
        sid = node.stabilizer_id
        if sid is None:
            continue
        if hasattr(node.operation, 'type') and node.operation.type == 'MS':
            if sid not in stab_qubits:
                stab_qubits[sid] = set()
            stab_qubits[sid].update(node.operation.targets)

    # ペアごとに左右に振り分け
    for i, (stab_a, stab_b) in enumerate(pairs):
        zone_a = MEMORY_LEFT if i % 2 == 0 else MEMORY_RIGHT
        zone_b = MEMORY_RIGHT if i % 2 == 0 else MEMORY_LEFT

        for qid in stab_qubits.get(stab_a, []):
            if qid not in initial:
                initial[qid] = zone_a
        for qid in stab_qubits.get(stab_b, []):
            if qid not in initial:
                initial[qid] = zone_b

    # 未割当のqubitがあればデフォルト配置
    for qid in data_qubits:
        if qid not in initial:
            initial[qid] = MEMORY_LEFT
    for qid in ancilla_qubits:
        if qid not in initial:
            initial[qid] = MEMORY_RIGHT

    return _enforce_capacity(initial)


def placement_greedy_alternate(data_qubits, ancilla_qubits, circuit_module):
    """貪欲法: 回路の登場順で左右交互に、LIZに近い端から配置

    量子回路を上から走査し、新しいqubitが登場するたびに
    左Memory → 右Memory → 左 → 右 ... と交互に割り当てる。
    並び順は両MemoryともLIZに近い端から積む:
      M1(z1, LIZ on right): 先頭に挿入 → 先に登場したqubitがLIZ側
      M2(z5, LIZ on left):  末尾に追加 → 先に登場したqubitがLIZ側
    """
    program = parse(circuit_module.CIRCUIT)
    assigned = set()
    left_order = []   # M1: 先頭に挿入するので逆順に構築
    right_order = []  # M2: 末尾に追加
    zones = [MEMORY_LEFT, MEMORY_RIGHT]
    zone_idx = 0

    for op in program.instructions:
        targets = []
        if hasattr(op, 'targets'):
            targets = op.targets
        for qid in targets:
            if qid not in assigned:
                assigned.add(qid)
                zone = zones[zone_idx]
                if zone == MEMORY_LEFT:
                    left_order.insert(0, qid)  # 先頭に挿入 → LIZ側(右端)
                else:
                    right_order.append(qid)     # 末尾に追加 → LIZ側(左端)
                zone_idx = 1 - zone_idx

    # 回路に登場しなかったqubit
    for qid in list(data_qubits) + list(ancilla_qubits):
        if qid not in assigned:
            assigned.add(qid)
            zone = zones[zone_idx]
            if zone == MEMORY_LEFT:
                left_order.insert(0, qid)
            else:
                right_order.append(qid)
            zone_idx = 1 - zone_idx

    # iteration orderが並び順になるようにdictを構築
    initial = {}
    for qid in left_order:
        initial[qid] = MEMORY_LEFT
    for qid in right_order:
        initial[qid] = MEMORY_RIGHT

    return _enforce_capacity(initial)


def _enforce_capacity(initial):
    """Memory容量を超過している場合、溢れたqubitを反対側に移動する"""
    left = [q for q, z in initial.items() if z == MEMORY_LEFT]
    right = [q for q, z in initial.items() if z == MEMORY_RIGHT]

    # 左が超過 → 超過分を右に移動（先頭=LIZから遠い側から移動）
    while len(left) > MEMORY_CAPACITY:
        q = left.pop(0)
        initial[q] = MEMORY_RIGHT
        right.append(q)

    # 右が超過 → 超過分を左に移動（末尾=LIZから遠い側から移動）
    while len(right) > MEMORY_CAPACITY:
        q = right.pop(-1)
        initial[q] = MEMORY_LEFT
        left.append(q)

    return initial


def placement_random(data_qubits, ancilla_qubits, rng: random.Random):
    """ランダム配置: 各qubitを左右どちらかにランダム割当"""
    initial = {}
    all_qubits = list(data_qubits) + list(ancilla_qubits)
    for qid in all_qubits:
        initial[qid] = rng.choice([MEMORY_LEFT, MEMORY_RIGHT])
    return _enforce_capacity(initial)


# ==========================================
# 実験ロジック
# ==========================================

def compile_with_placement(circuit_module_name: str, initial: dict,
                           interleave: bool = False, batch_size: int = 1,
                           memory_reorder: bool = False):
    """指定配置でコンパイル"""
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

    return {
        "total_ops": len(result.operations),
        "total_timesteps": len(result.timesteps),
        "seq_time_ms": seq_est.total_time / 1000,
        "par_time_ms": par_est.total_time / 1000,
        "stats": result.stats,
        "valid": validation.is_valid and par_validation.is_valid,
        "initial": initial,
        "result": result,
        "arch": arch,
    }


def describe_placement(initial: dict, data_qubits, ancilla_qubits):
    """配置の概要を返す（1行サマリ）"""
    left_data = [q for q in data_qubits if initial.get(q) == MEMORY_LEFT]
    right_data = [q for q in data_qubits if initial.get(q) == MEMORY_RIGHT]
    left_anc = [q for q in ancilla_qubits if initial.get(q) == MEMORY_LEFT]
    right_anc = [q for q in ancilla_qubits if initial.get(q) == MEMORY_RIGHT]
    return (f"L: {len(left_data)}D+{len(left_anc)}A={len(left_data)+len(left_anc)}, "
            f"R: {len(right_data)}D+{len(right_anc)}A={len(right_data)+len(right_anc)}")


def describe_placement_detail(initial: dict):
    """Memory内の並び順を可視化する

    initialのiteration orderがplace_ion_anyの追加順 = Memory内の並び順。
    zone 1 (左Memory) はLIZが右側、zone 5 (右Memory) はLIZが左側。
    """
    # iteration orderを保持してzone別に分類
    left_order = [ion for ion, zone in initial.items() if zone == MEMORY_LEFT]
    right_order = [ion for ion, zone in initial.items() if zone == MEMORY_RIGHT]
    lines = []
    lines.append(f"  M1(z1, LIZ on right): {left_order}")
    lines.append(f"  M2(z5, LIZ on left):  {right_order}")
    return "\n".join(lines)


def run_experiment(circuit_module_name: str, interleave: bool, random_samples: int, seed: int = 42):
    """1回路の配置比較実験"""

    circuit_module = importlib.import_module(circuit_module_name)
    data_qubits = circuit_module.DATA_QUBITS
    ancilla_qubits = circuit_module.ANCILLA_QUBITS

    print(f"\n{'=' * 70}")
    print(f"  {circuit_module_name} (interleave={'ON' if interleave else 'OFF'})")
    print(f"{'=' * 70}")

    # 定義済み配置
    placements = {
        "baseline": placement_baseline(data_qubits, ancilla_qubits),
        "split_even": placement_split_even(data_qubits, ancilla_qubits),
        "greedy_alt": placement_greedy_alternate(data_qubits, ancilla_qubits, circuit_module),
        "by_stabilizer": placement_by_stabilizer(data_qubits, ancilla_qubits, circuit_module),
    }

    results = {}
    for label, initial in placements.items():
        desc = describe_placement(initial, data_qubits, ancilla_qubits)
        try:
            r = compile_with_placement(circuit_module_name, initial, interleave=interleave)
            results[label] = r
            status = "PASS" if r["valid"] else "FAIL"
            print(f"  [{label:<16}] {desc}  → {r['par_time_ms']:.2f}ms  {status}")
            print(describe_placement_detail(initial))
        except Exception as e:
            print(f"  [{label:<16}] {desc}  → ERROR: {e}")

    # ランダム配置サンプリング
    if random_samples > 0:
        rng = random.Random(seed)
        random_all = []  # [(par_time_ms, result_dict), ...]
        print(f"\n  Random placement sampling ({random_samples} samples):")

        for i in range(random_samples):
            initial = placement_random(data_qubits, ancilla_qubits, rng)
            try:
                r = compile_with_placement(circuit_module_name, initial, interleave=interleave)
                random_all.append((r["par_time_ms"], r))
            except Exception as e:
                print(f"    Sample {i}: ERROR: {e}")

        if random_all:
            random_all.sort(key=lambda x: x[0])
            times = [t for t, _ in random_all]
            print(f"    Min:    {times[0]:.2f} ms")
            print(f"    Median: {times[len(times)//2]:.2f} ms")
            print(f"    Max:    {times[-1]:.2f} ms")
            print(f"    Mean:   {sum(times)/len(times):.2f} ms")

            # Top 3 を保存
            top_n = min(3, len(random_all))
            print(f"\n  Top {top_n} random placements:")
            for rank in range(top_n):
                t, r = random_all[rank]
                desc = describe_placement(r["initial"], data_qubits, ancilla_qubits)
                print(f"    #{rank+1}: {t:.2f} ms  {desc}")
                print(describe_placement_detail(r["initial"]))
                results[f"random_top{rank+1}"] = r

            results["random_stats"] = {
                "min": times[0],
                "max": times[-1],
                "mean": sum(times) / len(times),
                "median": times[len(times) // 2],
                "samples": len(times),
                "all_times": times,
            }

    # Summary table
    print()
    show_labels = [l for l in results.keys() if l != "random_stats"]
    print(f"  {'Strategy':<20} {'Ops':>8} {'TS':>8} {'Seq(ms)':>10} {'Par(ms)':>10} {'Valid':>6}")
    print(f"  {'-' * 62}")
    for label in show_labels:
        r = results[label]
        if isinstance(r, dict) and "total_ops" in r:
            valid = "PASS" if r["valid"] else "FAIL"
            print(f"  {label:<20} {r['total_ops']:>8} {r['total_timesteps']:>8} "
                  f"{r['seq_time_ms']:>10.2f} {r['par_time_ms']:>10.2f} {valid:>6}")

    # 最速の配置の命令列を txt で保存
    best_label = None
    best_time = float('inf')
    for label in show_labels:
        r = results[label]
        if isinstance(r, dict) and "par_time_ms" in r and r["par_time_ms"] < best_time:
            best_time = r["par_time_ms"]
            best_label = label

    if best_label and "result" in results[best_label]:
        output_dir = Path(__file__).parent.parent / 'results' / 'placement'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "interleave" if interleave else "sequential"
        txt_file = output_dir / f"{circuit_module_name}_best_{mode}_{timestamp}.txt"
        r = results[best_label]
        save_parallel_txt(
            r["result"].timesteps, str(txt_file),
            initial_placement=r["initial"], arch=r["arch"],
        )
        print(f"\n  [Best: {best_label}] {best_time:.2f} ms")
        print(f"  Saved: {txt_file.name}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Initial Placement Experiment')
    parser.add_argument('--circuit', default='snake_per_stabilizer',
                        help='Circuit module name')
    parser.add_argument('--interleave', action='store_true',
                        help='Enable interleave mode')
    parser.add_argument('--random-samples', type=int, default=RANDOM_SAMPLES,
                        help=f'Number of random placement samples (default: {RANDOM_SAMPLES})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--liz-capacity', type=int, default=LIZ_CAPACITY,
                        help=f'LIZ capacity (default: {LIZ_CAPACITY})')
    args = parser.parse_args()

    seed = args.seed

    # モジュールレベル設定を上書き
    import sys
    sys.modules[__name__].LIZ_CAPACITY = args.liz_capacity

    print("=" * 70)
    print("Initial Placement Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architecture: T-M(z1)-T-L(z3, cap={LIZ_CAPACITY})-T-M(z5)-T")
    print(f"Interleave: {'ON' if args.interleave else 'OFF'}")
    print(f"Random samples: {args.random_samples}")
    print("=" * 70)

    # Sequential mode
    results_seq = run_experiment(args.circuit, interleave=False, random_samples=args.random_samples, seed=seed)

    # Interleave mode (if requested)
    results_inter = None
    if args.interleave:
        results_inter = run_experiment(args.circuit, interleave=True, random_samples=args.random_samples, seed=seed)

    # Save JSON
    output_dir = Path(__file__).parent.parent / 'results' / 'placement'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON非対応のオブジェクトを除外するキー
    _skip_keys = {"result", "arch"}

    def placement_to_memory_order(initial):
        """initialのiteration orderからMemory内の並び順を生成"""
        left = [ion for ion, zone in initial.items() if zone == MEMORY_LEFT]
        right = [ion for ion, zone in initial.items() if zone == MEMORY_RIGHT]
        return {
            "M1_z1_LIZ_on_right": left,
            "M2_z5_LIZ_on_left": right,
        }

    def serialize_results(res):
        out = {}
        for label, r in res.items():
            if label == "random_stats":
                out[label] = r
            elif isinstance(r, dict) and "total_ops" in r:
                entry = {k: v for k, v in r.items() if k not in _skip_keys}
                if "initial" in r:
                    entry["memory_order"] = placement_to_memory_order(r["initial"])
                    entry["initial"] = {str(k): v for k, v in r["initial"].items()}
                out[label] = entry
        return out

    summary = {
        "experiment": "placement_comparison",
        "generated_at": datetime.now().isoformat(),
        "circuit": args.circuit,
        "architecture": {
            "type": "simple_mlm",
            "liz_capacity": LIZ_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
            "temp_capacity": TEMP_CAPACITY,
        },
        "sequential": serialize_results(results_seq),
    }
    if results_inter:
        summary["interleave"] = serialize_results(results_inter)

    json_file = output_dir / f"placement_comparison_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Output] Summary saved to: {json_file}")


if __name__ == "__main__":
    main()
