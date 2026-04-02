"""
Noise Injection Experiment

コンパイラのフォノン追跡結果を元に、Stim回路にゲートごとの
ノイズを挿入する。

パイプライン:
  1. 独自フォーマット回路 → コンパイル → MSゲートごとのフォノン数
  2. フォノン数 → エラー率 (ブラックボックスモジュール)
  3. Stim回路の各CXにDEPOLARIZE2を挿入 → ノイズ付きStim回路

使い方:
    python exp_noise_injection.py
    python exp_noise_injection.py --circuit snake_reuse1 --distance 3
    python exp_noise_injection.py --distance 5
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

import stim
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator


# ==========================================
# ブラックボックス: フォノン → エラー率
# ==========================================

def get_error_rate(axial_phonon: float, transversal_phonon: float) -> float:
    """
    フォノン数からMSゲートのエラー率を計算する。

    TODO: 他の人のモジュールに置き換え

    現在のplaceholder:
        基本エラー率 0.001 にフォノン数に比例した項を加算
    """
    base_error = 0.001
    # Placeholder: axial phonon 1000あたり +0.001 のエラー増加
    phonon_error = axial_phonon * 1e-6
    return min(base_error + phonon_error, 0.5)  # 上限50%


# ==========================================
# Step 1: コンパイル + フォノン取得
# ==========================================

def compile_with_phonon(circuit_module_name: str, placement: str = "split_even",
                        liz_capacity: int = 4, memory_capacity: int = 20):
    """
    独自フォーマット回路をコンパイルし、MSゲートごとのフォノン数を取得

    Returns:
        phonon_at_gates: [{'gate': 'MS(0,9)', 'phonons': {ion: (ax, tr)}}, ...]
    """
    mod = __import__(circuit_module_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()

    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=liz_capacity,
        memory_capacity=memory_capacity,
        temp_capacity=memory_capacity,
    )

    # split_even 配置
    initial = {}
    data = mod.DATA_QUBITS
    half = len(data) // 2
    for i, qid in enumerate(data):
        initial[qid] = 1 if i < half else 5
    for qid in mod.ANCILLA_QUBITS:
        initial[qid] = 5

    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial)

    est = ParallelEstimator()
    r = est.estimate(result.timesteps)

    return result, r, initial


# ==========================================
# Step 2: フォノン → エラー率リスト
# ==========================================

def phonon_to_error_rates(phonon_at_gates: list) -> list:
    """
    各MSゲートのフォノン数からエラー率を計算

    Returns:
        [(gate_name, error_rate, max_axial, max_transversal), ...]
    """
    error_rates = []
    for gate_info in phonon_at_gates:
        phonons = gate_info['phonons']
        # phonons: {ion_id: (axial, transversal)}
        # 各値はタプルまたはリスト
        max_ax = max(p[0] for p in phonons.values())
        max_tr = max(p[1] for p in phonons.values())
        err = get_error_rate(max_ax, max_tr)
        error_rates.append((gate_info['gate'], err, max_ax, max_tr))
    return error_rates


# ==========================================
# Step 3: Stim回路にノイズ挿入
# ==========================================

def generate_stim_circuit(distance: int, rounds: int = 1) -> stim.Circuit:
    """ノイズなしStim回路を生成"""
    return stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=rounds,
        distance=distance,
    )


def inject_noise(circuit: stim.Circuit, error_rates: list) -> stim.Circuit:
    """
    Stim回路の各CXゲートにDEPOLARIZE2を挿入

    CXの出現順とerror_ratesの順序が対応する前提。
    各CX命令には複数ペアが含まれうるので、ペア単位で展開して
    個別のエラー率を設定する。

    Args:
        circuit: ノイズなしStim回路
        error_rates: [(gate_name, error_rate, ax, tr), ...]

    Returns:
        ノイズ付きStim回路
    """
    noisy = stim.Circuit()
    error_idx = 0

    for instruction in circuit:
        if instruction.name == 'CX':
            # CXをペアごとに分解して個別エラー率を挿入
            targets = instruction.targets_copy()
            for i in range(0, len(targets), 2):
                if i + 1 >= len(targets):
                    break
                q1 = targets[i].value
                q2 = targets[i + 1].value

                # CXペアを追加
                noisy.append("CX", [q1, q2])

                # エラー率を取得して DEPOLARIZE2 を挿入
                if error_idx < len(error_rates):
                    _, err_rate, _, _ = error_rates[error_idx]
                    if err_rate > 0:
                        noisy.append("DEPOLARIZE2", [q1, q2], err_rate)
                    error_idx += 1
            # TICKはCXの後に元の回路で来るので、ここでは追加しない
        else:
            # CX以外はそのまま追加
            noisy.append(instruction)

    return noisy


# ==========================================
# メイン
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Noise Injection Experiment')
    parser.add_argument('--circuit', default='snake_reuse1',
                        help='Circuit module name (default: snake_reuse1)')
    parser.add_argument('--distance', type=int, default=3,
                        help='Code distance for Stim circuit (default: 3)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of syndrome measurement rounds (default: 1)')
    parser.add_argument('--memory-capacity', type=int, default=20,
                        help='Memory zone capacity (default: 20)')
    parser.add_argument('--output-dir', default='results/noise_injection',
                        help='Output directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Noise Injection Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Circuit: {args.circuit} (d={args.distance}, rounds={args.rounds})")
    print("=" * 70)

    # Step 1: コンパイル + フォノン
    print("\n[Step 1] Compiling with phonon tracking...")
    result, par_result, initial = compile_with_phonon(
        args.circuit,
        memory_capacity=args.memory_capacity,
    )
    pd = result.phonon_data

    print(f"  Total ops: {len(result.operations)}")
    print(f"  Parallel time: {par_result.total_time/1000:.2f} ms")
    print(f"  Max axial phonon: {pd['max_axial']:.2f}")
    print(f"  Max transversal phonon: {pd['max_transversal']:.2f}")
    print(f"  MS gates: {len(pd['at_gates'])}")

    # Step 2: フォノン → エラー率
    print("\n[Step 2] Computing error rates from phonon numbers...")
    error_rates = phonon_to_error_rates(pd['at_gates'])

    print(f"  Error rate range: {min(e[1] for e in error_rates):.6f} - {max(e[1] for e in error_rates):.6f}")
    print(f"  Mean error rate: {sum(e[1] for e in error_rates)/len(error_rates):.6f}")
    print()
    print(f"  {'Gate':15s} {'Error Rate':>12s} {'Axial':>10s} {'Trans':>10s}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
    for gate_name, err, ax, tr in error_rates:
        print(f"  {gate_name:15s} {err:12.6f} {ax:10.2f} {tr:10.4f}")

    # Step 3: Stim回路にノイズ挿入
    print(f"\n[Step 3] Generating noisy Stim circuit (d={args.distance})...")
    stim_circuit = generate_stim_circuit(args.distance, args.rounds)

    # CXペア数の確認
    stim_cx_count = sum(
        len(inst.targets_copy()) // 2
        for inst in stim_circuit
        if inst.name == 'CX'
    )
    print(f"  Stim CX pairs: {stim_cx_count}")
    print(f"  Compiler MS gates: {len(error_rates)}")

    if stim_cx_count != len(error_rates):
        print(f"  WARNING: CX count mismatch! Using min({stim_cx_count}, {len(error_rates)}) error rates")

    noisy_circuit = inject_noise(stim_circuit, error_rates)

    # 出力保存
    # .stim ファイル (Stimシミュレーション入力)
    stim_path = output_dir / f"noisy_d{args.distance}_{args.circuit}_{timestamp}.stim"
    with open(stim_path, 'w') as f:
        f.write(str(noisy_circuit))
    print(f"\n  Saved Stim circuit: {stim_path}")

    # JSON メタデータ
    json_path = output_dir / f"noisy_d{args.distance}_{args.circuit}_{timestamp}.json"
    metadata = {
        "generated_at": timestamp,
        "circuit_module": args.circuit,
        "distance": args.distance,
        "rounds": args.rounds,
        "compiler": {
            "total_ops": len(result.operations),
            "total_timesteps": len(result.timesteps),
            "par_time_ms": par_result.total_time / 1000,
            "seq_time_ms": par_result.sequential_time / 1000,
        },
        "phonon": {
            "max_axial": pd['max_axial'],
            "max_transversal": pd['max_transversal'],
        },
        "error_rates": [
            {"gate": name, "error_rate": err, "axial": ax, "transversal": tr}
            for name, err, ax, tr in error_rates
        ],
        "stim_cx_pairs": stim_cx_count,
        "stim_file": stim_path.name,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {json_path}")

    # Stim でデコード検証 (サンプル)
    print(f"\n[Step 4] Quick verification with Stim sampler...")
    try:
        sampler = noisy_circuit.compile_detector_sampler()
        shots = sampler.sample(shots=10000)
        detection_rate = shots.any(axis=1).mean()
        print(f"  Detection rate (10k shots): {detection_rate:.4f}")
        print(f"  (fraction of shots with at least one detection event)")
    except Exception as e:
        print(f"  Stim sampling failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
