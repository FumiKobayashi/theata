"""Smoke test for stim_recompiler.

Goal: take a QCCD compile result and verify that recompile_to_stim() produces
a Stim circuit with the expected structure.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stim_recompiler import recompile_to_stim


def main():
    parser = argparse.ArgumentParser(description='Recompile smoke test')
    parser.add_argument('--circuit', default='snake_reuse1',
                        help='Circuit module name (default: snake_reuse1)')
    parser.add_argument('--dump', metavar='PATH',
                        help='Save the recompiled Stim circuit to a file')
    args = parser.parse_args()

    print(f"Loading circuit: {args.circuit}")
    mod = __import__(args.circuit)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    print(f"  data qubits: {len(mod.DATA_QUBITS)}, ancillae: {mod.ANCILLA_QUBITS}")
    print(f"  DAG nodes: {len(dag.nodes)}")

    # memory capacity を回路サイズに合わせて確保（半分ずつ 2 ゾーンに置く前提）
    half = (len(mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    # Simple initial placement
    initial = {}
    for i, q in enumerate(mod.DATA_QUBITS):
        initial[q] = 1 if i < len(mod.DATA_QUBITS) // 2 else 5
    for a in mod.ANCILLA_QUBITS:
        initial[a] = 5

    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial)

    print(f"\nCompile result:")
    print(f"  TimeSteps: {len(result.timesteps)}")
    print(f"  flat ops: {len(result.flat_ops)}")
    print(f"  DAG attached: {result.dag is not None}")
    print(f"  initial_placement attached: {result.initial_placement is not None}")

    # Recompile to Stim
    print(f"\nRecompiling to Stim...")
    try:
        stim_circuit = recompile_to_stim(result)
    except Exception as e:
        print(f"[FAIL] Recompile raised: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"[OK] Recompile succeeded")
    print(f"  Stim circuit num_qubits: {stim_circuit.num_qubits}")
    print(f"  Stim instructions: {len([_ for _ in stim_circuit])}")

    # Count op types
    op_counts = {}
    for inst in stim_circuit:
        op_counts[inst.name] = op_counts.get(inst.name, 0) + 1
    print(f"  Op breakdown:")
    for name in sorted(op_counts):
        print(f"    {name}: {op_counts[name]}")

    # Optional dump
    if args.dump:
        out_path = Path(args.dump)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(str(stim_circuit))
        print(f"\n  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
