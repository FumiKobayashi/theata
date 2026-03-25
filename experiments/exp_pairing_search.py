"""
Stabilizer Pairing × Placement Search Experiment

d=3 rotated surface code (8 stabilizers) に対して、
スタビライザのペアリング × 初期配置 × LIZ容量の全探索を行う。

探索空間:
  - ペアリング: 8スタビライザから4ペアの組み方 = 105通り
  - 配置: split_even + greedy_alt + ランダムサンプル
  - LIZ容量: 4, 6 (コマンドラインで指定)

目的:
  interleaveが有効になるペアリング×配置の組み合わせを発見する。
  特に「共有qubitが少なく左右分離が良いペア」が有利かを検証。

使い方:
    python exp_pairing_search.py
    python exp_pairing_search.py --liz-capacity 6
    python exp_pairing_search.py --liz-capacity 6 --random-samples 10
    python exp_pairing_search.py --top 10
"""

import sys
import argparse
import random
import json
import itertools
from pathlib import Path
from datetime import datetime
from collections import Counter

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

CIRCUIT_MODULE = "snake_per_stabilizer"
MEMORY_LEFT = 1
MEMORY_RIGHT = 5
MEMORY_CAPACITY = 20
TEMP_CAPACITY = 20


# ==========================================
# ペアリング列挙
# ==========================================

def enumerate_pairings(items):
    """
    itemsから2つずつペアを作る全組み合わせを列挙。
    items=[A,B,C,D] → [(A,B),(C,D)], [(A,C),(B,D)], [(A,D),(B,C)]
    """
    if len(items) <= 1:
        yield []
        return
    first = items[0]
    rest = items[1:]
    for i, partner in enumerate(rest):
        pair = (first, partner)
        remaining = rest[:i] + rest[i+1:]
        for sub_pairing in enumerate_pairings(remaining):
            yield [pair] + sub_pairing


def get_stabilizer_info(circuit_module):
    """回路からスタビライザ情報を抽出"""
    program = parse(circuit_module.CIRCUIT)
    stabilizers = []
    current_ms_targets = []

    for op in program.instructions:
        if isinstance(op, GateOp) and op.type == 'MS':
            current_ms_targets.append(tuple(op.targets))
        elif isinstance(op, MeasureOp):
            ancilla = op.targets[0]
            data_qubits = set()
            for pair in current_ms_targets:
                for q in pair:
                    if q != ancilla:
                        data_qubits.add(q)
            stabilizers.append({
                'ancilla': ancilla,
                'data_qubits': sorted(data_qubits),
            })
            current_ms_targets = []

    return stabilizers


def pairing_shared_qubits(pairing, stabilizers):
    """ペアリングの共有qubit数の合計を返す"""
    stab_map = {s['ancilla']: set(s['data_qubits']) for s in stabilizers}
    total_shared = 0
    for a, b in pairing:
        shared = stab_map[a] & stab_map[b]
        total_shared += len(shared)
    return total_shared


def pairing_description(pairing, stabilizers):
    """ペアリングの説明文"""
    stab_map = {s['ancilla']: set(s['data_qubits']) for s in stabilizers}
    parts = []
    for a, b in pairing:
        shared = sorted(stab_map[a] & stab_map[b])
        shared_str = f"shared={shared}" if shared else "shared=none"
        parts.append(f"(a{a},a{b} {shared_str})")
    return " ".join(parts)


# ==========================================
# 配置生成
# ==========================================

def placement_split_even(data_qubits, ancilla_qubits, pairing=None):
    """均等分散配置（アンシラをLIZ近傍に配置）"""
    initial = {}
    half = len(data_qubits) // 2
    sorted_data = sorted(data_qubits)
    for i, qid in enumerate(sorted_data):
        if i < half:
            initial[qid] = MEMORY_LEFT
        else:
            initial[qid] = MEMORY_RIGHT

    # アンシラ: ペアごとに左右に分ける（ペアリング情報がある場合）
    if pairing and len(ancilla_qubits) >= 2:
        # ペアの最初のアンシラを左、2番目を右
        assigned = set()
        left_count = 0
        right_count = 0
        for a, b in pairing:
            if a in ancilla_qubits and a not in assigned:
                if left_count <= right_count:
                    initial[a] = MEMORY_LEFT
                    left_count += 1
                else:
                    initial[a] = MEMORY_RIGHT
                    right_count += 1
                assigned.add(a)
            if b in ancilla_qubits and b not in assigned:
                # 相方は反対側に
                if initial.get(a) == MEMORY_LEFT:
                    initial[b] = MEMORY_RIGHT
                    right_count += 1
                else:
                    initial[b] = MEMORY_LEFT
                    left_count += 1
                assigned.add(b)
        # 未割当アンシラ
        for qid in ancilla_qubits:
            if qid not in assigned:
                initial[qid] = MEMORY_LEFT if left_count <= right_count else MEMORY_RIGHT
    else:
        half_a = len(ancilla_qubits) // 2
        for i, qid in enumerate(sorted(ancilla_qubits)):
            if i < half_a:
                initial[qid] = MEMORY_LEFT
            else:
                initial[qid] = MEMORY_RIGHT

    return initial


def placement_by_pairing(data_qubits, ancilla_qubits, pairing, stabilizers):
    """ペアリングに基づく配置: 各ペアのデータqubitを左右に分散"""
    stab_map = {s['ancilla']: set(s['data_qubits']) for s in stabilizers}
    initial = {}

    for pair_idx, (a, b) in enumerate(pairing):
        # アンシラ: ペア内で左右に分ける
        if pair_idx % 2 == 0:
            initial[a] = MEMORY_LEFT
            initial[b] = MEMORY_RIGHT
        else:
            initial[a] = MEMORY_RIGHT
            initial[b] = MEMORY_LEFT

        # データqubit: Aのデータ→Aと同側、Bのデータ→Bと同側
        a_only = stab_map[a] - stab_map[b]
        b_only = stab_map[b] - stab_map[a]
        shared = stab_map[a] & stab_map[b]

        for q in a_only:
            if q not in initial:
                initial[q] = initial[a]
        for q in b_only:
            if q not in initial:
                initial[q] = initial[b]
        # 共有qubit: 均等に分ける
        for i, q in enumerate(sorted(shared)):
            if q not in initial:
                initial[q] = MEMORY_LEFT if i % 2 == 0 else MEMORY_RIGHT

    # 未割当を均等に
    left_count = sum(1 for v in initial.values() if v == MEMORY_LEFT)
    right_count = sum(1 for v in initial.values() if v == MEMORY_RIGHT)
    for q in sorted(set(data_qubits) | set(ancilla_qubits)):
        if q not in initial:
            if left_count <= right_count:
                initial[q] = MEMORY_LEFT
                left_count += 1
            else:
                initial[q] = MEMORY_RIGHT
                right_count += 1

    return initial


def placement_random(data_qubits, ancilla_qubits, rng):
    """ランダム配置"""
    initial = {}
    all_qubits = list(data_qubits) + list(ancilla_qubits)
    for qid in all_qubits:
        initial[qid] = rng.choice([MEMORY_LEFT, MEMORY_RIGHT])
    return initial


# ==========================================
# コンパイル実行
# ==========================================

def compile_with_pairing(circuit_module_name, initial, pairing, liz_capacity,
                         interleave=True, batch_size=2):
    """指定ペアリング・配置でコンパイル"""
    circuit_module = __import__(circuit_module_name)
    program = parse(circuit_module.CIRCUIT)

    if interleave:
        builder = DAGBuilder(program, stabilizer_parallel=True)
        dag = builder.build()
        # カスタムペアリングでhook error edges追加
        builder.add_hook_error_edges(pairs=pairing)
    else:
        builder = DAGBuilder(program, stabilizer_parallel=False)
        dag = builder.build()

    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=liz_capacity,
        memory_capacity=MEMORY_CAPACITY,
        temp_capacity=TEMP_CAPACITY,
    )

    config = CompilerConfig(
        interleave=interleave,
        measure_batch_size=batch_size if interleave else 1,
        memory_reorder=True,
    )
    compiler = QCCDCompiler(arch, config=config)

    try:
        result = compiler.compile(dag, initial)
    except Exception as e:
        return None, str(e)

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
    }, None


# ==========================================
# メイン実験
# ==========================================

def run_experiment(liz_capacity, random_samples, top_n, seed):
    circuit_module = __import__(CIRCUIT_MODULE)
    stabilizers = get_stabilizer_info(circuit_module)
    ancilla_ids = [s['ancilla'] for s in stabilizers]
    data_qubits = circuit_module.DATA_QUBITS
    ancilla_qubits = circuit_module.ANCILLA_QUBITS

    print("=" * 70)
    print(f"Stabilizer Pairing × Placement Search")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Circuit: {CIRCUIT_MODULE}")
    print(f"LIZ capacity: {liz_capacity}")
    print(f"Stabilizers: {len(stabilizers)}")
    print(f"Random placement samples per pairing: {random_samples}")
    print("=" * 70)

    # スタビライザ情報
    print("\nStabilizer structure:")
    for i, s in enumerate(stabilizers):
        print(f"  S{i}: ancilla={s['ancilla']}, data={s['data_qubits']}")

    # ペアリング列挙
    all_pairings = list(enumerate_pairings(ancilla_ids))
    print(f"\nTotal pairings: {len(all_pairings)}")

    # まず baseline (interleave OFF, default pairing) を取得
    print("\n--- Baseline (interleave=OFF, split_even) ---")
    baseline_initial = placement_split_even(data_qubits, ancilla_qubits)
    baseline_result, err = compile_with_pairing(
        CIRCUIT_MODULE, baseline_initial, None, liz_capacity,
        interleave=False, batch_size=1
    )
    if baseline_result:
        print(f"  par_time: {baseline_result['par_time_ms']:.2f} ms  "
              f"ops: {baseline_result['total_ops']}  "
              f"valid: {baseline_result['valid']}")
        baseline_time = baseline_result['par_time_ms']
    else:
        print(f"  ERROR: {err}")
        baseline_time = None

    # 全ペアリングを探索
    print(f"\n--- Searching {len(all_pairings)} pairings × placements (interleave=ON) ---\n")

    results = []
    rng = random.Random(seed)

    for p_idx, pairing in enumerate(all_pairings):
        shared = pairing_shared_qubits(pairing, stabilizers)
        desc = pairing_description(pairing, stabilizers)

        placements = {}

        # 配置1: split_even (ペアリング考慮)
        placements['split_even'] = placement_split_even(
            data_qubits, ancilla_qubits, pairing
        )

        # 配置2: by_pairing
        placements['by_pairing'] = placement_by_pairing(
            data_qubits, ancilla_qubits, pairing, stabilizers
        )

        # 配置3-N: ランダム
        for r in range(random_samples):
            placements[f'random_{r}'] = placement_random(
                data_qubits, ancilla_qubits, rng
            )

        best_time = float('inf')
        best_placement_name = None
        best_result = None
        best_initial = None

        for pl_name, initial in placements.items():
            result, err = compile_with_pairing(
                CIRCUIT_MODULE, initial, pairing, liz_capacity,
                interleave=True, batch_size=2
            )
            if result and result['valid']:
                if result['par_time_ms'] < best_time:
                    best_time = result['par_time_ms']
                    best_placement_name = pl_name
                    best_result = result
                    best_initial = initial

        if best_result:
            results.append({
                'pairing': pairing,
                'shared_qubits': shared,
                'description': desc,
                'best_placement': best_placement_name,
                'best_time': best_time,
                'best_initial': best_initial,
                'best_result': best_result,
            })

        # 進捗表示
        if (p_idx + 1) % 10 == 0 or p_idx == 0 or p_idx == len(all_pairings) - 1:
            status = f"{best_time:.2f}ms" if best_result else "FAIL"
            print(f"  [{p_idx+1:>3}/{len(all_pairings)}] shared={shared} "
                  f"best={status} ({best_placement_name})  {desc}")

    # ソートして上位を表示
    results.sort(key=lambda r: r['best_time'])

    print(f"\n{'=' * 70}")
    print(f"Top {top_n} pairings (interleave=ON, LIZ={liz_capacity}):")
    print(f"{'=' * 70}")
    if baseline_time:
        print(f"  Baseline (interleave=OFF): {baseline_time:.2f} ms\n")

    for rank, r in enumerate(results[:top_n]):
        speedup = ""
        if baseline_time:
            diff = r['best_time'] - baseline_time
            pct = diff / baseline_time * 100
            speedup = f"  vs baseline: {diff:+.2f}ms ({pct:+.1f}%)"

        print(f"  #{rank+1}: {r['best_time']:.2f} ms  "
              f"shared={r['shared_qubits']}  "
              f"placement={r['best_placement']}{speedup}")
        print(f"       {r['description']}")

        # 配置詳細
        initial = r['best_initial']
        left = sorted([q for q, z in initial.items() if z == MEMORY_LEFT])
        right = sorted([q for q, z in initial.items() if z == MEMORY_RIGHT])
        left_d = [q for q in left if q in data_qubits]
        left_a = [q for q in left if q in ancilla_qubits]
        right_d = [q for q in right if q in data_qubits]
        right_a = [q for q in right if q in ancilla_qubits]
        print(f"       M1(z1): {left} ({len(left_d)}D+{len(left_a)}A)")
        print(f"       M2(z5): {right} ({len(right_d)}D+{len(right_a)}A)")
        print(f"       ops={r['best_result']['total_ops']}  "
              f"ts={r['best_result']['total_timesteps']}  "
              f"valid={r['best_result']['valid']}")
        print()

    # worst表示
    if len(results) > top_n:
        worst = results[-1]
        print(f"  Worst: {worst['best_time']:.2f} ms  "
              f"shared={worst['shared_qubits']}  {worst['description']}")

    # 共有qubit数と実行時間の関係
    print(f"\n--- Shared qubits vs best time ---")
    by_shared = {}
    for r in results:
        s = r['shared_qubits']
        if s not in by_shared:
            by_shared[s] = []
        by_shared[s].append(r['best_time'])
    for s in sorted(by_shared):
        times = by_shared[s]
        print(f"  shared={s}: min={min(times):.2f}ms  "
              f"max={max(times):.2f}ms  "
              f"mean={sum(times)/len(times):.2f}ms  "
              f"count={len(times)}")

    # JSON保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / 'results' / 'pairing_search'
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"pairing_search_{timestamp}.json"

    save_data = {
        "metadata": {
            "circuit": CIRCUIT_MODULE,
            "liz_capacity": liz_capacity,
            "random_samples": random_samples,
            "total_pairings": len(all_pairings),
            "timestamp": timestamp,
            "baseline_time_ms": baseline_time,
        },
        "top_results": [],
        "shared_qubit_analysis": {},
    }

    for r in results[:top_n * 2]:
        save_data["top_results"].append({
            "pairing": [(a, b) for a, b in r['pairing']],
            "shared_qubits": r['shared_qubits'],
            "description": r['description'],
            "best_placement": r['best_placement'],
            "best_time_ms": r['best_time'],
            "initial": r['best_initial'],
            "total_ops": r['best_result']['total_ops'],
            "total_timesteps": r['best_result']['total_timesteps'],
            "stats": r['best_result']['stats'],
            "valid": r['best_result']['valid'],
        })

    for s in sorted(by_shared):
        times = by_shared[s]
        save_data["shared_qubit_analysis"][str(s)] = {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "count": len(times),
        }

    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Output] Saved to: {json_path}")

    # 最速のtxt出力
    if results:
        best = results[0]
        txt_path = results_dir / f"best_pairing_LIZ{liz_capacity}_{timestamp}.txt"
        save_parallel_txt(
            best['best_result']['result'].timesteps,
            str(txt_path),
        )
        print(f"[Output] Best result saved to: {txt_path}")


# ==========================================
# エントリポイント
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stabilizer pairing search")
    parser.add_argument("--liz-capacity", type=int, default=6,
                        help="LIZ capacity (default: 6)")
    parser.add_argument("--random-samples", type=int, default=5,
                        help="Random placement samples per pairing (default: 5)")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N results (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_experiment(
        liz_capacity=args.liz_capacity,
        random_samples=args.random_samples,
        top_n=args.top,
        seed=args.seed,
    )
