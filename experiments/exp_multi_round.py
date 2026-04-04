"""
Multi-Round QEC Simulation

複数ラウンドのシンドローム測定をシミュレーションする。
奇数ラウンドはforward走査、偶数ラウンドはreverse走査を使用。

ラウンド間の配置引き継ぎ:
  - 各ラウンド終了時のion配置を次ラウンドの初期配置として使用
  - フォノン数もラウンド間で蓄積（冷却なし仮定）

使い方:
    python exp_multi_round.py
    python exp_multi_round.py --rounds 10
    python exp_multi_round.py --placement greedy_alt --rounds 5
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator


# ==========================================
# 設定
# ==========================================

# forward / reverse 回路モジュール名
FORWARD_CIRCUIT = "snake_reuse1"
REVERSE_CIRCUIT = "snake_reuse1_reverse"

LIZ_CAPACITY = 4
MEMORY_CAPACITY = 20


# ==========================================
# 配置生成
# ==========================================

def placement_split_even(data_qubits, ancilla_qubits):
    """均等分散: データを左右半分ずつ、アンシラはLIZ近傍"""
    initial = {}
    half = len(data_qubits) // 2
    for i, qid in enumerate(data_qubits):
        initial[qid] = 1 if i < half else 5
    for qid in ancilla_qubits:
        initial[qid] = 5
    return initial


def placement_greedy_alt(data_qubits, ancilla_qubits, circuit_module):
    """貪欲法: 回路の登場順で左右交互に、LIZに近い端から配置"""
    from qccd_parser import parse as _parse
    program = _parse(circuit_module.CIRCUIT)
    assigned = set()
    left_order = []
    right_order = []
    zones = [1, 5]
    zone_idx = 0

    for op in program.instructions:
        targets = []
        if hasattr(op, 'targets'):
            targets = op.targets
        for qid in targets:
            if qid not in assigned:
                assigned.add(qid)
                zone = zones[zone_idx]
                if zone == 1:
                    left_order.insert(0, qid)
                else:
                    right_order.append(qid)
                zone_idx = 1 - zone_idx

    for qid in list(data_qubits) + list(ancilla_qubits):
        if qid not in assigned:
            assigned.add(qid)
            zone = zones[zone_idx]
            if zone == 1:
                left_order.insert(0, qid)
            else:
                right_order.append(qid)
            zone_idx = 1 - zone_idx

    initial = {}
    for qid in left_order:
        initial[qid] = 1
    for qid in right_order:
        initial[qid] = 5
    return initial


# ==========================================
# 1ラウンドのコンパイル
# ==========================================

def compile_round(circuit_module_name, initial_placement, arch,
                  initial_zone_contents=None, eviction_strategy="balanced"):
    """
    1ラウンドをコンパイルし、結果を返す。

    initial_zone_contents が指定されている場合、zone内の並び順を保持して配置する。
    フォノン数はそのラウンド単体での値（0から開始）。
    """
    mod = __import__(circuit_module_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()

    # zone_contents の順序を保持した initial_placement を構築
    if initial_zone_contents is not None:
        ordered_placement = {}
        for zone_id in sorted(initial_zone_contents.keys()):
            for ion_id in initial_zone_contents[zone_id]:
                ordered_placement[ion_id] = zone_id
        initial_placement = ordered_placement

    config = CompilerConfig(eviction_strategy=eviction_strategy)
    compiler = QCCDCompiler(arch, config=config)
    result = compiler.compile(dag, initial_placement)

    est = ParallelEstimator()
    par_result = est.estimate(result.timesteps)

    return result, par_result


# ==========================================
# メイン
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Round QEC Simulation')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of QEC rounds (default: 5)')
    parser.add_argument('--placement', default='split_even',
                        choices=['split_even', 'greedy_alt'],
                        help='Initial placement strategy (default: split_even)')
    parser.add_argument('--forward-circuit', default=FORWARD_CIRCUIT,
                        help=f'Forward circuit module (default: {FORWARD_CIRCUIT})')
    parser.add_argument('--reverse-circuit', default=REVERSE_CIRCUIT,
                        help=f'Reverse circuit module (default: {REVERSE_CIRCUIT})')
    parser.add_argument('--memory-capacity', type=int, default=MEMORY_CAPACITY,
                        help=f'Memory zone capacity (default: {MEMORY_CAPACITY})')
    parser.add_argument('--eviction', default='balanced',
                        choices=['balanced', 'nearest'],
                        help='Eviction strategy (default: balanced)')
    parser.add_argument('--output-dir', default='results/multi_round',
                        help='Output directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-Round QEC Simulation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rounds: {args.rounds}")
    print(f"Placement: {args.placement}")
    forward_circuit = args.forward_circuit
    reverse_circuit = args.reverse_circuit
    eviction_strategy = args.eviction
    mem_cap = args.memory_capacity
    print(f"Forward: {forward_circuit}, Reverse: {reverse_circuit}")
    print(f"Eviction: {eviction_strategy}")
    print("=" * 70)

    # Architecture
    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=LIZ_CAPACITY,
        memory_capacity=mem_cap,
        temp_capacity=mem_cap,
    )

    # Initial placement
    fwd_mod = __import__(forward_circuit)
    if args.placement == 'split_even':
        initial = placement_split_even(fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS)
    elif args.placement == 'greedy_alt':
        initial = placement_greedy_alt(fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    print(f"\nInitial placement: {initial}")

    # Multi-round simulation
    rounds_data = []
    current_placement = initial
    current_zone_contents = None  # Round 1 は zone_contents なし（placement dict 順）
    total_par_time = 0.0
    total_seq_time = 0.0

    for r in range(1, args.rounds + 1):
        direction = "forward" if r % 2 == 1 else "reverse"
        circuit_name = forward_circuit if direction == "forward" else reverse_circuit

        print(f"\n--- Round {r}/{args.rounds} ({direction}) ---")

        result, par_result = compile_round(
            circuit_name, current_placement, arch,
            initial_zone_contents=current_zone_contents,
            eviction_strategy=eviction_strategy,
        )

        pd = result.phonon_data
        total_par_time += par_result.total_time
        total_seq_time += par_result.sequential_time

        # 最終配置の zone_contents 表示
        fzc = result.final_zone_contents or {}
        for z in sorted(fzc.keys()):
            print(f"  zone {z}: {fzc[z]}")

        print(f"  Ops: {len(result.operations)}, TS: {len(result.timesteps)}")
        print(f"  Par time: {par_result.total_time/1000:.2f} ms")
        print(f"  Cumulative par time: {total_par_time/1000:.2f} ms")
        print(f"  Max axial phonon (this round): {pd['max_axial']:.2f}")
        print(f"  Max transversal phonon (this round): {pd['max_transversal']:.2f}")

        # ラウンドデータ記録（フォノンはそのラウンド単体の値）
        round_info = {
            "round": r,
            "direction": direction,
            "circuit": circuit_name,
            "initial_placement": {str(k): v for k, v in current_placement.items()},
            "final_placement": {str(k): v for k, v in result.final_placement.items()},
            "final_zone_contents": {str(z): ions for z, ions in (result.final_zone_contents or {}).items()},
            "total_ops": len(result.operations),
            "total_timesteps": len(result.timesteps),
            "par_time_ms": par_result.total_time / 1000,
            "seq_time_ms": par_result.sequential_time / 1000,
            "phonon_max_axial": pd['max_axial'],
            "phonon_max_transversal": pd['max_transversal'],
            "phonon_at_gates": pd['at_gates'],
        }
        rounds_data.append(round_info)

        # 次ラウンドへの引き継ぎ: 配置 + zone内の並び順
        current_placement = result.final_placement
        current_zone_contents = result.final_zone_contents

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"\n  {'Round':>5} {'Dir':>8} {'Par(ms)':>10} {'Cum(ms)':>10} {'Ops':>6} {'MaxAx':>10} {'MaxTr':>8}")
    print(f"  {'-'*58}")

    cum = 0.0
    for rd in rounds_data:
        cum += rd['par_time_ms']
        print(f"  {rd['round']:>5} {rd['direction']:>8} {rd['par_time_ms']:>10.2f} "
              f"{cum:>10.2f} {rd['total_ops']:>6} "
              f"{rd['phonon_max_axial']:>10.2f} {rd['phonon_max_transversal']:>8.2f}")

    avg_time = total_par_time / args.rounds / 1000
    print(f"\n  Average par time per round: {avg_time:.2f} ms")
    print(f"  Total par time ({args.rounds} rounds): {total_par_time/1000:.2f} ms")

    # JSON出力
    output = {
        "experiment": "multi_round_qec",
        "generated_at": timestamp,
        "forward_circuit": forward_circuit,
        "reverse_circuit": reverse_circuit,
        "distance": 3,
        "total_rounds": args.rounds,
        "scan_pattern": "alternating",
        "placement_strategy": args.placement,
        "eviction_strategy": eviction_strategy,
        "architecture": {
            "type": "simple_mlm",
            "liz_capacity": LIZ_CAPACITY,
            "memory_capacity": MEMORY_CAPACITY,
        },
        "initial_placement": {str(k): v for k, v in initial.items()},
        "summary": {
            "total_par_time_ms": total_par_time / 1000,
            "total_seq_time_ms": total_seq_time / 1000,
            "avg_par_time_per_round_ms": avg_time,
            "max_single_round_phonon_axial": max(rd['phonon_max_axial'] for rd in rounds_data),
            "max_single_round_phonon_transversal": max(rd['phonon_max_transversal'] for rd in rounds_data),
        },
        "rounds": rounds_data,
    }

    json_path = output_dir / f"multi_round_{args.placement}_{args.rounds}r_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Output] Saved to: {json_path}")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
