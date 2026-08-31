"""任意のスタビライザ測定順序からの回路生成 (d5_generator.generate_circuit_from_order)。

Usage:
    python tests/test_circuit_from_order.py
"""

import random
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent
for p in ('src', 'circuits', 'experiments'):
    sys.path.insert(0, str(ROOT / p))

import d5_generator as gen


def test_from_order_reproduces_snake_circuit():
    """蛇行順を渡すと従来の generate_circuit と同一の回路文字列になる"""
    for d in (3, 5):
        snake = gen.get_snake_order(gen.get_stabilizers(d), d)
        for reverse in (False, True):
            ref, ref_ordered, ref_anc = gen.generate_circuit(d, 1, reverse=reverse)
            out, out_ordered, out_anc = gen.generate_circuit_from_order(
                d, snake, 1, reverse=reverse, cnot_order="explicit")
            assert out == ref
            assert out_ordered == ref_ordered and out_anc == ref_anc
    print("test_from_order_reproduces_snake_circuit: OK")


def test_surface_cnot_resolution():
    """support だけ (昇順) を渡しても hook-error 回避の CNOT 順で埋まる"""
    d = 5
    ref = {frozenset(dq): (t, list(dq)) for t, dq, _ in gen.get_stabilizers(d)}
    supports_only = [(t, sorted(dq)) for t, dq, _ in gen.get_stabilizers(d)]
    resolved = gen.resolve_cnot_order(d, supports_only, cnot_order="surface")
    for stype, dq, pos in resolved:
        assert (stype, dq) == ref[frozenset(dq)], (stype, dq)
        assert pos is None
    # explicit は与えた順をそのまま使う
    explicit = gen.resolve_cnot_order(d, supports_only, cnot_order="explicit")
    assert [dq for _, dq, _ in explicit] == [dq for _, dq in supports_only]
    # 存在しない support / 型違いはエラー
    for bad in ([("X", [0, 1, 2, 3])], [("Z", [0, 1, 5, 6])]):
        try:
            gen.resolve_cnot_order(d, bad, cnot_order="surface")
            assert False, f"should raise for {bad}"
        except ValueError:
            pass
    print("test_surface_cnot_resolution: OK")


def test_permuted_order_yields_valid_syndrome_circuit():
    """ランダムな測定順序でも forward/reverse の 2 ラウンドがコンパイル・
    検証・Stim 再変換を通り、無ノイズで detector が全て 0 になる"""
    from qccd_parser import parse
    from dag_builder import DAGBuilder
    from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
    from validator import Validator
    from stim_recompiler import recompile_multi_round
    from exp_multi_round import placement_greedy_alt

    d = 3
    supports = [(t, sorted(dq)) for t, dq, _ in gen.get_stabilizers(d)]
    random.Random(7).shuffle(supports)

    fwd_str, ordered, anc = gen.generate_circuit_from_order(d, supports, 1, reverse=False)
    rev_str, _, _ = gen.generate_circuit_from_order(d, supports, 1, reverse=True)
    fwd = gen.make_circuit_module("_test_order_fwd", d, fwd_str, anc, ordered)
    rev = gen.make_circuit_module("_test_order_rev", d, rev_str, anc)
    assert __import__("_test_order_fwd") is fwd  # sys.modules 登録の確認

    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=20)
    placement = placement_greedy_alt(fwd.DATA_QUBITS, fwd.ANCILLA_QUBITS, fwd)
    results = []
    zone_contents = None
    for mod in (fwd, rev):
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=CompilerConfig()).compile(dag, placement)
        v = Validator(arch).validate(cr.flat_ops, placement)
        assert v.is_valid, v.summary()
        results.append(cr)
        placement, zone_contents = cr.final_placement, cr.final_zone_contents

    circuit = recompile_multi_round(results, observable_column=[0, 1, 2])
    det, obs = circuit.compile_detector_sampler().sample(200, separate_observables=True)
    assert det.sum() == 0 and obs.sum() == 0, "noiseless detectors non-zero"
    n_meas = sum(1 for op in results[0].flat_ops if getattr(op, 'type', '') == 'MEASURE')
    assert n_meas == len(supports)
    print("test_permuted_order_yields_valid_syndrome_circuit: OK")


if __name__ == "__main__":
    test_from_order_reproduces_snake_circuit()
    test_surface_cnot_resolution()
    test_permuted_order_yields_valid_syndrome_circuit()
    print("\nAll circuit-from-order tests passed.")
