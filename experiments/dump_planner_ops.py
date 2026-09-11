"""planner v1 / v2 が出力する命令列 (flat_ops) をテキストダンプする。

正準設定 (simple_mlm + greedy_alt 配置) で 1 ラウンドコンパイルし、
formatter.save_flat_ops_txt で 1 op / 1 行のテキストを保存する。
v2 では測定窓に emit される先行整列 / staging を目視確認できる。

Usage:
    python experiments/dump_planner_ops.py                    # snake_reuse1, v1+v2
    python experiments/dump_planner_ops.py d5_reuse1 --planners v2
    python experiments/dump_planner_ops.py --horizon 4 --prefetch-depth 1
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from formatter import save_flat_ops_txt
from exp_multi_round import placement_greedy_alt


def compile_one(name, cfg):
    mod = __import__(name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    half = (len(mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
    return QCCDCompiler(arch, config=cfg).compile(dag, initial), arch, initial


def main():
    ap = argparse.ArgumentParser(description='dump planner flat_ops')
    ap.add_argument('circuits', nargs='*', default=['snake_reuse1'])
    ap.add_argument('--planners', nargs='*', default=['v1', 'v2'],
                    choices=['v1', 'v2'])
    ap.add_argument('--horizon', type=int, default=2)
    ap.add_argument('--prefetch-depth', type=int, default=0)
    ap.add_argument('--out-dir', default='results/planner_v2/dumps')
    args = ap.parse_args()

    out_dir = Path(__file__).parent.parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.circuits:
        for planner in args.planners:
            if planner == 'v2':
                cfg = CompilerConfig(planner='v2',
                                     prefetch_depth=args.prefetch_depth,
                                     v2_sort_horizon=args.horizon)
                tag = f"v2_h{args.horizon}d{args.prefetch_depth}"
            else:
                cfg = CompilerConfig()
                tag = "v1"
            cr, arch, initial = compile_one(name, cfg)
            path = out_dir / f"{name}_{tag}_flat_ops.txt"
            save_flat_ops_txt(cr.flat_ops, str(path),
                              initial_placement=initial, arch=arch)
            print(f"{name} {tag}: {len(cr.flat_ops)} ops -> {path}")


if __name__ == '__main__':
    main()
