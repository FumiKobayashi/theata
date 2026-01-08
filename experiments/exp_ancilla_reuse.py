"""
Ancilla Reuse Experiment

比較:
1. Baseline (8 ancilla): surface_d3.py
2. 4 ancilla reuse (X→Z): surface_d3_reuse4.py
3. 4 ancilla reuse (interleaved): surface_d3_reuse4_interleaved.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from estimator import Estimator

# 回路モジュールをインポート
import surface_d3
import surface_d3_reuse4
import surface_d3_reuse4_interleaved


def run_experiment(circuit_module, arch, initial_placement, config, stabilizer_parallel=True):
    """単一の実験を実行"""
    program = parse(circuit_module.CIRCUIT)
    builder = DAGBuilder(program, stabilizer_parallel=stabilizer_parallel)
    dag = builder.build()
    
    compiler = QCCDCompiler(arch, config=config)
    result = compiler.compile(dag, initial_placement)
    estimation = Estimator().estimate(result.operations)
    
    return {
        'input_instructions': len(program.instructions),
        'output_operations': len(result.operations),
        'time_us': estimation.total_time,
        'time_ms': estimation.total_time / 1000,
        'counts': estimation.counts,
    }


def main():
    print("=" * 70)
    print("Ancilla Reuse Experiment")
    print("=" * 70)
    
    # アーキテクチャ設定
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=20)
    print(f"\nArchitecture: {arch}")
    
    # コンパイラ設定（baseline）
    config = CompilerConfig()
    print(f"Config: {config}")
    
    # 実験定義
    experiments = [
        {
            'name': '8 ancilla baseline (X to Z)',
            'module': surface_d3,
            'initial': {**{i: 0 for i in range(9)}, **{i: 2 for i in range(9, 17)}},
            'stabilizer_parallel': True,
        },
        {
            'name': '4 ancilla reuse (X to Z)',
            'module': surface_d3_reuse4,
            'initial': {**{i: 0 for i in range(9)}, **{i: 2 for i in range(9, 13)}},
            'stabilizer_parallel': True,
        },
        {
            'name': '4 ancilla reuse (interleaved)',
            'module': surface_d3_reuse4_interleaved,
            'initial': {**{i: 0 for i in range(9)}, **{i: 2 for i in range(9, 13)}},
            'stabilizer_parallel': True,
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n[{exp['name']}]")
        print(f"  Qubits: {exp['module'].QUBIT_COUNT}")
        print(f"  Data: {exp['module'].DATA_QUBITS}")
        print(f"  Ancilla: {exp['module'].ANCILLA_QUBITS}")
        
        result = run_experiment(
            exp['module'], 
            arch, 
            exp['initial'], 
            config, 
            exp['stabilizer_parallel']
        )
        
        print(f"  Input instructions: {result['input_instructions']}")
        print(f"  Output operations: {result['output_operations']}")
        print(f"  Time: {result['time_ms']:.2f} ms")
        print(f"  Reset: {result['counts'].get('Reset', 0)}, Measure: {result['counts'].get('Measure', 0)}")
        
        results.append({
            'name': exp['name'],
            'qubit_count': exp['module'].QUBIT_COUNT,
            'ancilla_count': len(exp['module'].ANCILLA_QUBITS),
            **result,
        })
    
    # サマリー表示
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    baseline_time = results[0]['time_ms']
    
    print(f"\n{'Config':<35} {'Ops':>8} {'Time(ms)':>10} {'Ancilla':>8} {'vs Base':>10}")
    print("-" * 75)
    
    for r in results:
        change = (r['time_ms'] / baseline_time - 1) * 100
        print(f"{r['name']:<35} {r['output_operations']:>8} {r['time_ms']:>10.2f} {r['ancilla_count']:>8} {change:>+10.1f}%")
    
    # JSON出力
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ancilla_reuse_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'architecture': str(arch),
        'config': config.to_dict(),
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Output] Saved to: {output_file}")


if __name__ == "__main__":
    main()
