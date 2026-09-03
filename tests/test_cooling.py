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


def test_auto_cool_threshold():
    """閾値ベース自動冷却: MS 直前挿入 + ゲート時フォノンが閾値以下になる"""
    from qccd_compiler import CompilerConfig

    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 9 AT (5, 0)

    RESET 9
    MS 0 9 0.25
    MS 1 9 0.25
    MEASURE 9
    """
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 9: 5}

    def compile_with(threshold):
        dag = DAGBuilder(parse(code)).build()
        cfg = CompilerConfig(cool_threshold_transversal=threshold)
        return QCCDCompiler(arch, config=cfg).compile(dag, dict(initial))

    # 閾値なし → CoolOp なし。輸送で transversal が僅かに載る
    base = compile_with(None)
    assert base.stats["cools"] == 0
    at_gates = base.phonon_data['at_gates']
    max_tr_base = max(max(p[1] for p in g['phonons'].values()) for g in at_gates)
    assert max_tr_base > 0.05, "テスト前提: 輸送で transversal が载ること"

    # 極小閾値 → MS 直前に CoolOp が挿入され、ゲート時 transversal は残留値 0
    thr = 0.05
    cooled = compile_with(thr)
    assert cooled.stats["cools"] > 0
    for g in cooled.phonon_data['at_gates']:
        max_tr = max(p[1] for p in g['phonons'].values())
        assert max_tr <= thr + 1e-9, f"gate {g['gate']} tr={max_tr} > {thr}"
    # CoolOp は対応する MS の直前に並ぶ
    flat = cooled.flat_ops
    for i, op in enumerate(flat):
        if isinstance(op, CoolOp):
            nxt = flat[i + 1]
            assert getattr(nxt, 'type', '') == 'MS'
            assert nxt.zone_id == op.zone_id
    # validator も通る
    v = Validator(arch).validate(cooled.flat_ops, initial)
    assert v.is_valid, v.summary()
    print("test_auto_cool_threshold: OK")


def test_cool_ancilla_after_measure():
    """アンシラ測定後冷却: MEASURE 直後に指定 duration の CoolOp が入る"""
    from qccd_compiler import CompilerConfig
    from parallel_estimator import ParallelEstimator

    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 9 AT (5, 0)

    RESET 9
    MS 0 9 0.25
    MEASURE 9
    RESET 9
    MS 1 9 0.25
    MEASURE 9
    """
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 9: 5}
    dag = DAGBuilder(parse(code)).build()
    cfg = CompilerConfig(cool_ancilla_after_measure_us=2000.0)
    result = QCCDCompiler(arch, config=cfg).compile(dag, dict(initial))

    # MEASURE 2 回 → CoolOp 2 個、各 duration=2000、直前が MEASURE
    cools = [(i, op) for i, op in enumerate(result.flat_ops)
             if isinstance(op, CoolOp)]
    assert len(cools) == 2, cools
    for i, op in cools:
        assert op.duration_us == 2000.0
        from ast_def import MeasureOp
        assert isinstance(result.flat_ops[i - 1], MeasureOp)
        assert result.flat_ops[i - 1].zone_id == op.zone_id

    # アンシラのフォノンは最終冷却後 0 (最後の op が MEASURE→COOL のため)
    assert result.phonon_data['per_ion_axial'][9] == 0.0
    assert result.phonon_data['per_ion_transversal'][9] == 0.0

    # estimator が per-op duration (2000) で計上する
    est = ParallelEstimator().estimate(result.timesteps)
    assert est.counts.get("Cool") == 2
    assert abs(est.breakdown["Cool"] - 4000.0) < 1e-9

    # validator も通る
    v = Validator(arch).validate(result.flat_ops, initial)
    assert v.is_valid, v.summary()
    print("test_cool_ancilla_after_measure: OK")


def test_cool_ancilla_measure_interval():
    """アンシラ測定後冷却の周期指定: MEASURE N 回につき CoolOp 1 個"""
    from ast_def import MeasureOp
    from qccd_compiler import CompilerConfig

    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 9 AT (5, 0)

    RESET 9
    MS 0 9 0.25
    MEASURE 9
    RESET 9
    MS 1 9 0.25
    MEASURE 9
    """
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 9: 5}

    def compile_with(interval):
        dag = DAGBuilder(parse(code)).build()
        cfg = CompilerConfig(cool_ancilla_after_measure_us=1000.0,
                             cool_ancilla_measure_interval=interval)
        return QCCDCompiler(arch, config=cfg).compile(dag, dict(initial))

    # interval=2: MEASURE 2 回 → CoolOp は 2 回目の直後の 1 個だけ
    result = compile_with(2)
    flat = result.flat_ops
    cools = [(i, op) for i, op in enumerate(flat) if isinstance(op, CoolOp)]
    assert len(cools) == 1, cools
    i, op = cools[0]
    assert op.duration_us == 1000.0
    assert isinstance(flat[i - 1], MeasureOp)
    n_meas_before = sum(1 for o in flat[:i] if isinstance(o, MeasureOp))
    assert n_meas_before == 2, "CoolOp は 2 回目の MEASURE の後にあるべき"

    # interval=1 (デフォルト) は従来通り毎回
    assert compile_with(1).stats["cools"] == 2

    # interval=3: 測定 2 回では一度も冷却されない
    assert compile_with(3).stats["cools"] == 0

    # validation: 0 以下や非 int はエラー
    try:
        CompilerConfig(cool_ancilla_measure_interval=0).validate()
        assert False, "interval=0 should raise"
    except ValueError as e:
        assert "cool_ancilla_measure_interval" in str(e)
    print("test_cool_ancilla_measure_interval: OK")


def test_merge_cool_memory_unit():
    """merge 平均化冷却の単体: 冷アンシラの合流で memory チェーンが冷える"""
    from ast_def import MeasureOp, SplitOp, ShuttleOp, MergeOp
    from qccd_compiler import CompilerConfig, MovementPlanner

    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    state = QCCDState(architecture=arch)
    state.phonon_params = phonon_params_from()

    # memory 1: 熱いデータチェーン [0, 1] (tr=3.0)
    # memory 5: 冷たいチェーン [2, 3] (tr=0.1、閾値以下 → 対象外)
    # LIZ 3: 測定直後のアンシラ 9 (温まっている → 冷却命令が入る)
    for ion, zone, tr, ax in [(0, 1, 3.0, 50.0), (1, 1, 3.0, 50.0),
                              (2, 5, 0.1, 1.0), (3, 5, 0.1, 1.0),
                              (9, 3, 0.5, 5.0)]:
        if arch.is_memory(zone):
            state.place_ion(ion, zone)
        else:  # LIZ への直接配置 (測定直後の状況を再現)
            state.zone_contents.setdefault(zone, []).append(ion)
            state.ion_locations[ion] = zone
        state.phonon_transversal[ion] = tr
        state.phonon_axial[ion] = ax

    planner = MovementPlanner(
        arch, CompilerConfig(merge_cool_threshold_transversal=1.0))
    ops = planner._maybe_merge_cool_memory(
        MeasureOp(targets=[9], zone_id=3), state)

    # 冷却 2 回 (合流前の直接冷却 + 帰還後の再冷却) + 往復搬送
    cools = [op for op in ops if isinstance(op, CoolOp)]
    assert len(cools) == 2 and all(c.zone_id == 3 for c in cools)
    assert isinstance(ops[0], CoolOp), "冷却が搬送より先"
    assert isinstance(ops[-1], CoolOp), "帰還後に再冷却"
    assert any(isinstance(op, SplitOp) for op in ops)
    assert any(isinstance(op, ShuttleOp) for op in ops)
    assert any(isinstance(op, MergeOp) for op in ops)

    # アンシラは LIZ に帰還、memory 1 のデータ順は保存される
    assert state.locate(9) == 3
    assert state.zone_contents[1] == [0, 1]

    # memory 1 は平均化で冷えている: (3+3+0.1)/3 + merge 0.05 ≈ 2.13
    tr0 = state.phonon_transversal[0]
    tr1 = state.phonon_transversal[1]
    assert abs(tr0 - tr1) < 1e-9
    assert 2.0 < tr0 < 2.2, f"expected ~2.13, got {tr0}"

    # 閾値以下の memory 5 は手つかず
    assert state.phonon_transversal[2] == 0.1
    assert state.phonon_transversal[3] == 0.1
    assert state.zone_contents[5] == [2, 3]

    # 全 memory が閾値以下なら何も挿入しない
    planner2 = MovementPlanner(
        arch, CompilerConfig(merge_cool_threshold_transversal=10.0))
    assert planner2._maybe_merge_cool_memory(
        MeasureOp(targets=[9], zone_id=3), state) == []

    # validation: 負の閾値はエラー
    try:
        CompilerConfig(merge_cool_threshold_transversal=-1.0).validate()
        assert False, "negative threshold should raise"
    except ValueError as e:
        assert "merge_cool_threshold_transversal" in str(e)
    print("test_merge_cool_memory_unit: OK")


def test_merge_cool_integration():
    """merge 平均化冷却の統合: MEASURE 後に往復搬送が入り validator を通る"""
    from ast_def import MeasureOp, ShuttleOp
    from qccd_compiler import CompilerConfig

    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 9 AT (5, 0)

    RESET 9
    MS 0 9 0.25
    MEASURE 9
    """
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 9: 5}

    # 極小閾値: 測定前の退避で温まった memory チェーンが必ず発動させる
    dag = DAGBuilder(parse(code)).build()
    cfg = CompilerConfig(merge_cool_threshold_transversal=0.01)
    result = QCCDCompiler(arch, config=cfg).compile(dag, dict(initial))
    flat = result.flat_ops

    assert result.stats["cools"] >= 1
    meas_idx = max(i for i, op in enumerate(flat) if isinstance(op, MeasureOp))
    tail = flat[meas_idx + 1:]
    # MEASURE 後: アンシラ 9 が memory へ行き、LIZ (3) へ戻る shuttle がある
    mem_zones = {z.id for z in arch.get_memory_zones()}
    visits = [op for op in tail if isinstance(op, ShuttleOp)
              and op.target == 9 and op.dst.trap in mem_zones]
    returns = [op for op in tail if isinstance(op, ShuttleOp)
               and op.target == 9 and op.dst.trap == 3]
    assert visits, "ancilla should visit a memory zone after MEASURE"
    assert returns, "ancilla should return to the LIZ"
    assert flat.index(returns[-1]) > flat.index(visits[0])
    assert result.final_zone_contents.get(3) == [9]

    # validator (sequential + parallel) を通る
    v = Validator(arch).validate(flat, initial)
    assert v.is_valid, v.summary()
    pv = ParallelValidator(arch).validate(result.timesteps, initial)
    assert pv.is_valid, pv.summary()

    # 無効時 (None) は往復搬送なし
    dag2 = DAGBuilder(parse(code)).build()
    base = QCCDCompiler(arch, config=CompilerConfig()).compile(
        dag2, dict(initial))
    base_meas = max(i for i, op in enumerate(base.flat_ops)
                    if isinstance(op, MeasureOp))
    assert not any(isinstance(op, ShuttleOp) and op.target == 9
                   for op in base.flat_ops[base_meas + 1:])
    print("test_merge_cool_integration: OK")


if __name__ == "__main__":
    test_sim_params_loading()
    test_parse_cool()
    test_cool_chain_unit()
    test_pipeline_with_cool()
    test_cool_with_coolant_transport()
    test_auto_cool_threshold()
    test_cool_ancilla_after_measure()
    test_cool_ancilla_measure_interval()
    test_merge_cool_memory_unit()
    test_merge_cool_integration()
    print("\nAll cooling tests passed.")
