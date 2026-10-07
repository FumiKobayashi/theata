"""LIZ 常駐 coolant (resident_coolant_ids + coolant_recool_us) のテスト。

2026-10-06 先生コメントの構成: LIZ に coolant を常駐させ、データ量子
ビットがゲート訪問のたびに冷たい常駐チェーンとの merge 平均化で冷える。
常駐イオンは回路外で、全退避から保護され、測定・リセット中も同居を許す。

Usage:
    python tests/test_resident_coolant.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ast_def import CoolOp, ShuttleOp, SplitOp
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from validator import Validator

CODE = """
QUBIT 0 AT (1, 0)
QUBIT 1 AT (1, 1)
QUBIT 2 AT (1, 2)
QUBIT 3 AT (1, 3)
QUBIT 8 AT (1, 4)
QUBIT 9 AT (1, 5)
MS 0 8 0.25
MS 1 8 0.25
MEASURE 8
MS 2 9 0.25
MS 3 9 0.25
MEASURE 9
"""
COOLANTS = [9000, 9001]
LIZ = 3


def _compile(cfg):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(CODE)).build()
    initial = {0: 1, 1: 1, 2: 1, 3: 1, 8: 1, 9: 1}
    for c in COOLANTS:
        initial[c] = LIZ
    comp = QCCDCompiler(arch, config=cfg)
    res = comp.compile(dag, dict(initial))
    return res, initial


def test_resident_stays_and_validates():
    """常駐 coolant は一度も LIZ を離れず、スケジュールは validator を通る"""
    cfg = CompilerConfig(resident_coolant_ids=COOLANTS,
                         coolant_recool_us=400.0,
                         cool_ancilla_after_measure_us=2000.0)
    res, initial = _compile(cfg)

    # 最終配置: coolant は LIZ のまま
    for c in COOLANTS:
        assert res.final_placement[c] == LIZ

    # coolant を動かす輸送 op が存在しない
    for op in res.flat_ops:
        if isinstance(op, ShuttleOp):
            assert op.target not in COOLANTS

    v = Validator(QCCDArchitecture.simple_mlm()).validate(res.flat_ops, initial)
    assert v.is_valid, v.summary()
    print("test_resident_stays_and_validates: OK")


def test_passive_cooling_lowers_gate_phonons():
    """常駐 coolant との merge 平均化でゲート時フォノンが下がり、
    補助一斉再冷却 (coolant_recool_us) の CoolOp が挿入される"""
    base_cfg = CompilerConfig(cool_ancilla_after_measure_us=2000.0)
    res_base, _ = _compile(base_cfg)
    # ベースラインでは coolant 配置だけして保護しない状況は作れないため、
    # 「常駐なし (coolant を置かない)」と比較する
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(CODE)).build()
    initial = {0: 1, 1: 1, 2: 1, 3: 1, 8: 1, 9: 1}
    res_nores = QCCDCompiler(arch, config=base_cfg).compile(dag, dict(initial))

    res_cfg = CompilerConfig(resident_coolant_ids=COOLANTS,
                             coolant_recool_us=400.0,
                             cool_ancilla_after_measure_us=2000.0)
    res_res, _ = _compile(res_cfg)

    def max_gate_tr(res):
        return max(p[1] for g in res.phonon_data['at_gates']
                   for p in g['phonons'].values())
    assert max_gate_tr(res_res) < max_gate_tr(res_nores)

    n_cool_res = sum(1 for op in res_res.flat_ops if isinstance(op, CoolOp))
    n_cool_base = sum(1 for op in res_nores.flat_ops if isinstance(op, CoolOp))
    assert n_cool_res > n_cool_base, "一斉再冷却の CoolOp が入るはず"
    print("test_passive_cooling_lowers_gate_phonons: OK")


def test_data_actuator_moves_data_not_ancilla():
    """データ移動型 (predictive_cool_actuator='data'): エピソードで動くのは
    予測超過データイオンのみで、帰還後も配置整合・排他ルールを守る"""
    from qccd_compiler import QCCDCompiler as _QC
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(CODE)).build()
    initial = {0: 1, 1: 1, 2: 1, 3: 1, 8: 1, 9: 1}
    for c in COOLANTS:
        initial[c] = LIZ
    cfg = CompilerConfig(resident_coolant_ids=COOLANTS,
                         coolant_recool_us=400.0,
                         predictive_cool_threshold_transversal=0.0,
                         predictive_cool_actuator='data',
                         cool_ancilla_after_measure_us=2000.0)
    comp = _QC(arch, config=cfg)
    res = comp.compile(dag, dict(initial))

    episodes = comp._planner.episodes
    assert episodes, "データ訪問エピソードが発動するはず"
    ep_ids = {i for ep in episodes for i in ep}
    ep_ops = [op for op in res.flat_ops if id(op) in ep_ids]
    shuttled = {op.target for op in ep_ops if isinstance(op, ShuttleOp)}
    data_ions = {0, 1, 2, 3}
    assert shuttled, "エピソード内にデータの Shuttle があるはず"
    assert shuttled <= data_ions, \
        f"エピソードで動いて良いのはデータのみ: {shuttled}"

    # 排他ルール (Rule A/B) と validator
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cooling_schedule_rules import check_cooling_schedule_rules
    rep = check_cooling_schedule_rules(res.timesteps, episodes)
    assert rep['same_ts'] == [], rep['same_ts']
    assert rep['span'] == [], rep['span']

    v = Validator(arch).validate(res.flat_ops, initial)
    assert v.is_valid, v.summary()
    print(f"test_data_actuator_moves_data_not_ancilla: OK "
          f"({len(episodes)} visits, moved={sorted(shuttled)})")


def test_no_resident_is_backward_compatible():
    """resident 無効 (既定) では op 列が従来と完全一致"""
    arch = QCCDArchitecture.simple_mlm()
    initial = {0: 1, 1: 1, 2: 1, 3: 1, 8: 1, 9: 1}
    r1 = QCCDCompiler(arch, config=CompilerConfig()).compile(
        DAGBuilder(parse(CODE)).build(), dict(initial))
    r2 = QCCDCompiler(arch, config=CompilerConfig(
        resident_coolant_ids=None, coolant_recool_us=None)).compile(
        DAGBuilder(parse(CODE)).build(), dict(initial))
    assert len(r1.flat_ops) == len(r2.flat_ops)
    assert all(type(a) is type(b) for a, b in zip(r1.flat_ops, r2.flat_ops))
    print("test_no_resident_is_backward_compatible: OK")


if __name__ == "__main__":
    test_resident_stays_and_validates()
    test_passive_cooling_lowers_gate_phonons()
    test_data_actuator_moves_data_not_ancilla()
    test_no_resident_is_backward_compatible()
    print("\nAll resident coolant tests passed.")
