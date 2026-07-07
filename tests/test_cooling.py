"""Cooling 命令 (CoolOp) + パラメータ外部化 (sim_params) のテスト。

Usage:
    python tests/test_cooling.py
"""

import json
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ast_def import CoolOp
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, QCCDState
from estimator import Estimator, TimeParameters
from parallel_estimator import ParallelEstimator
from validator import Validator, ParallelValidator
from sim_params import (
    load_time_parameters, phonon_params_from, dump_time_parameters,
)


def test_sim_params_loading():
    """デフォルト / ファイル / override / 未知キーの動作確認"""
    # デフォルト
    tp = load_time_parameters()
    assert tp.ms_gate == 59.0
    assert tp.cool == 400.0

    # ファイル + override (override が勝つ)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "custom.json"
        p.write_text(json.dumps({"ms_gate": 100.0, "cool": 250.0}),
                     encoding='utf-8')
        tp2 = load_time_parameters(p, overrides={"cool": 300.0})
        assert tp2.ms_gate == 100.0
        assert tp2.cool == 300.0
        assert tp2.r_gate == 4.0  # 未指定はデフォルト

        # dump → load の往復
        p3 = Path(d) / "dumped.json"
        dump_time_parameters(tp2, p3)
        tp3 = load_time_parameters(p3)
        assert tp3 == tp2

    # 未知キーはエラー
    try:
        load_time_parameters(overrides={"ms_gate_typo": 1.0})
        assert False, "unknown key should raise"
    except ValueError as e:
        assert "ms_gate_typo" in str(e)

    # phonon dict 形式の一貫性 (旧 DEFAULT_PHONON_PARAMS と同値)
    ph = phonon_params_from()
    assert ph['split_axial'] == 10.0
    assert ph['shuttle_base_distance'] == 1000.0
    assert ph['cool_residual_axial'] == 0.0
    print("test_sim_params_loading: OK")


def test_parse_cool():
    """DSL の COOL 命令 (coolant あり / なし)"""
    prog = parse("""
    QUBIT 0 AT (1, 0)
    COOL 1
    COOL 3 0
    """)
    ops = prog.instructions
    assert len(ops) == 2
    assert isinstance(ops[0], CoolOp)
    assert ops[0].zone_id == 1 and ops[0].coolant_id == -1
    assert ops[1].zone_id == 3 and ops[1].coolant_id == 0
    print("test_parse_cool: OK")


def test_cool_chain_unit():
    """QCCDState.cool_chain: 残留値までの低減 (増えないこと含む)"""
    arch = QCCDArchitecture.simple_mlm()
    state = QCCDState(architecture=arch)
    state.phonon_params = phonon_params_from(
        TimeParameters(phonon_cool_residual_axial=0.5,
                       phonon_cool_residual_transversal=0.1))
    state.place_ion(0, 1)
    state.place_ion(1, 1)
    state.phonon_axial[0] = 20.0
    state.phonon_transversal[0] = 1.0
    state.phonon_axial[1] = 0.2      # 残留値より低い → そのまま
    state.phonon_transversal[1] = 0.05

    state.cool_chain(1)
    assert state.phonon_axial[0] == 0.5
    assert state.phonon_transversal[0] == 0.1
    assert state.phonon_axial[1] == 0.2
    assert state.phonon_transversal[1] == 0.05
    print("test_cool_chain_unit: OK")


def test_pipeline_with_cool():
    """DSL → DAG → compile → validate → estimate の統合確認。

    MS ゲート後に LIZ (zone 3) を冷却する。COOL は DAG 上バリアなので
    必ず MS の後に実行され、LIZ 上の 2 イオンのフォノンが残留値に落ちる。
    """
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)

    R 0 1.57 0.0
    MS 0 1 0.25
    COOL 3
    """
    program = parse(code)
    dag = DAGBuilder(program).build()

    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1}

    residual = 0.25
    tp = load_time_parameters(overrides={
        "cool": 500.0,
        "phonon_cool_residual_axial": residual,
        "phonon_cool_residual_transversal": residual,
    })
    compiler = QCCDCompiler(arch, time_params=tp)
    result = compiler.compile(dag, initial)

    # COOL が出力に含まれ、stats に計上される
    cools = [op for op in result.flat_ops if isinstance(op, CoolOp)]
    assert len(cools) == 1
    assert result.stats["cools"] == 1

    # COOL は MS より後
    ops = result.flat_ops
    ms_idx = max(i for i, op in enumerate(ops)
                 if getattr(op, 'type', '') == 'MS')
    cool_idx = ops.index(cools[0])
    assert cool_idx > ms_idx

    # LIZ 上のイオン (gate 後は zone 3 に居る) のフォノンが残留値以下
    final_zone3 = result.final_zone_contents.get(3, [])
    assert set(final_zone3) == {0, 1}, f"ions not in LIZ: {result.final_zone_contents}"
    for ion in final_zone3:
        assert result.phonon_data['per_ion_axial'][ion] <= residual + 1e-12
        assert result.phonon_data['per_ion_transversal'][ion] <= residual + 1e-12

    # 冷却なしと比較してフォノンが実際に下がっている
    dag2 = DAGBuilder(parse(code.replace("COOL 3\n", ""))).build()
    result2 = QCCDCompiler(arch, time_params=tp).compile(dag2, dict(initial))
    assert result2.phonon_data['max_axial'] > residual

    # validator (sequential + parallel) を通る
    v = Validator(arch).validate(result.flat_ops, initial)
    assert v.is_valid, v.summary()
    pv = ParallelValidator(arch).validate(result.timesteps, initial)
    assert pv.is_valid, pv.summary()

    # estimator が Cool を duration=500 で計上する
    est = ParallelEstimator(tp).estimate(result.timesteps)
    assert est.breakdown.get("Cool") == 500.0
    assert est.counts.get("Cool") == 1
    est_seq = Estimator(tp).estimate(result.flat_ops)
    assert est_seq.breakdown.get("Cool") == 500.0
    print("test_pipeline_with_cool: OK")


def test_cool_with_coolant_transport():
    """coolant 指定時: coolant がゾーン外に居ればコンパイラが搬送する"""
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 5 AT (5, 0)

    MS 0 1 0.25
    COOL 3 5
    """
    program = parse(code)
    dag = DAGBuilder(program).build()
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 5: 5}

    result = QCCDCompiler(arch).compile(dag, initial)
    cools = [op for op in result.flat_ops if isinstance(op, CoolOp)]
    assert len(cools) == 1 and cools[0].coolant_id == 5
    # COOL 実行時点で coolant 5 は zone 3 に搬送済みのはず → validator で検証
    v = Validator(arch).validate(result.flat_ops, initial)
    assert v.is_valid, v.summary()
    # coolant も冷却対象チェーンに居るので残留値 (デフォルト 0) になっている
    assert result.phonon_data['per_ion_axial'][5] == 0.0
    print("test_cool_with_coolant_transport: OK")


if __name__ == "__main__":
    test_sim_params_loading()
    test_parse_cool()
    test_cool_chain_unit()
    test_pipeline_with_cool()
    test_cool_with_coolant_transport()
    print("\nAll cooling tests passed.")
