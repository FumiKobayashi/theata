"""予測型 merge 冷却 (predictive_cool_threshold_transversal) のテスト。

2026-10-02 プロトコル: 2-pass コンパイル。
  pass 1 = 本トリガのみ無効で測定窓ごとの per-ion 増分 Δtr を記録
  pass 2 = 各 MEASURE 直後に「現在値 + 次窓 Δ > θ」のデータ量子ビットが
           居る memory へ merge 冷却エピソードを挿入

Usage:
    python tests/test_predictive_cooling.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ast_def import CoolOp, MeasureOp
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from validator import Validator

# 2 スタビライザ: 窓1 = (data 0,1 × anc 8)、窓2 = (data 2,3 × anc 9)
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
INITIAL = {0: 1, 1: 1, 2: 1, 3: 1, 8: 1, 9: 1}


def _compile(cfg):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(CODE)).build()
    comp = QCCDCompiler(arch, config=cfg)
    res = comp.compile(dag, dict(INITIAL))
    return comp, res


def test_window_table_structure():
    """pass 1 の窓テーブル: 測定イベントごとに 1 窓、ancilla/データ集合が正しい"""
    # θ を高くして発動なし → テーブル構造だけ検証
    comp, res = _compile(CompilerConfig(
        predictive_cool_threshold_transversal=1000.0))
    table = comp._planner.window_table
    assert table is not None and len(table) == 2
    assert table[0]['ancillas'] == [8]
    assert table[1]['ancillas'] == [9]
    assert table[0]['data_ions'] == {0, 1}
    assert table[1]['data_ions'] == {2, 3}
    # 窓2 のデータイオンは窓2 で輸送される → Δ > 0
    assert all(table[1]['delta_tr'][i] > 0 for i in (2, 3))
    print("test_window_table_structure: OK")


def test_no_fire_means_identical_ops():
    """θ 超過なし (高閾値) なら op 列はトリガ無効時と完全一致"""
    _, res_off = _compile(CompilerConfig())
    _, res_hi = _compile(CompilerConfig(
        predictive_cool_threshold_transversal=1000.0))
    assert len(res_off.flat_ops) == len(res_hi.flat_ops)
    assert all(type(a) is type(b) for a, b in
               zip(res_off.flat_ops, res_hi.flat_ops))
    assert res_off.phonon_data['per_ion_transversal'] == \
        res_hi.phonon_data['per_ion_transversal']
    print("test_no_fire_means_identical_ops: OK")


def test_predictive_fires_before_window():
    """低 θ: 窓2 の予測超過 → MEASURE 8 の直後 (窓2 開始前) に冷却が入り、
    窓2 のゲート時フォノンが下がる"""
    _, res_off = _compile(CompilerConfig())
    comp, res_on = _compile(CompilerConfig(
        predictive_cool_threshold_transversal=0.05))
    assert comp._planner.window_table is not None, "整合検証で無効化されていない"

    cools = [op for op in res_on.flat_ops if isinstance(op, CoolOp)]
    assert cools, "予測超過で CoolOp が入るはず"

    # 挿入位置: 最初の CoolOp は MEASURE 8 の後・MEASURE 9 の前
    idx_m8 = next(i for i, op in enumerate(res_on.flat_ops)
                  if isinstance(op, MeasureOp) and 8 in op.targets)
    idx_m9 = next(i for i, op in enumerate(res_on.flat_ops)
                  if isinstance(op, MeasureOp) and 9 in op.targets)
    idx_cool = next(i for i, op in enumerate(res_on.flat_ops)
                    if isinstance(op, CoolOp))
    assert idx_m8 < idx_cool < idx_m9

    # 窓2 のゲート (MS(2,9), MS(3,9)) のデータイオン tr が低下
    def w2_gate_tr(res):
        vals = []
        for g in res.phonon_data['at_gates']:
            if g['gate'] in ('MS(2,9)', 'MS(3,9)'):
                for ion in (2, 3):
                    if ion in g['phonons']:
                        vals.append(g['phonons'][ion][1])
        return max(vals)
    assert w2_gate_tr(res_on) < w2_gate_tr(res_off)

    # validator 通過
    arch = QCCDArchitecture.simple_mlm()
    v = Validator(arch).validate(res_on.flat_ops, dict(INITIAL))
    assert v.is_valid, v.summary()
    print("test_predictive_fires_before_window: OK")


def test_max_iters_repeats_episode():
    """θ を床以下にすると max_iters 回までエピソードが繰り返される"""
    def n_cools(max_iters):
        _, res = _compile(CompilerConfig(
            predictive_cool_threshold_transversal=0.0,
            predictive_cool_max_iters=max_iters))
        return res.stats.get('cools', 0)
    c1, c3 = n_cools(1), n_cools(3)
    assert c3 > c1, f"max_iters=3 ({c3}) should insert more cools than 1 ({c1})"
    print("test_max_iters_repeats_episode: OK")


def test_prediction_matches_actual_when_no_insertion():
    """挿入が起きない窓では、予測値 = pass 1 絶対値 = pass 2 実測値 (厳密性)"""
    comp, res = _compile(CompilerConfig(
        predictive_cool_threshold_transversal=1000.0))
    table = comp._planner.window_table
    # 発動なし → pass 2 の軌跡は pass 1 と同一。窓境界で Δ を足し上げた値が
    # 最終 per-ion 値と一致するはず
    final = res.phonon_data['per_ion_transversal']
    acc = {ion: 0.0 for ion in final}
    for w in table:
        for ion, dv in w['delta_tr'].items():
            acc[ion] += dv
    for ion in final:
        assert abs(acc[ion] - final[ion]) < 1e-9, \
            f"ion {ion}: sum(delta)={acc[ion]} != final={final[ion]}"
    print("test_prediction_matches_actual_when_no_insertion: OK")


if __name__ == "__main__":
    test_window_table_structure()
    test_no_fire_means_identical_ops()
    test_predictive_fires_before_window()
    test_max_iters_repeats_episode()
    test_prediction_matches_actual_when_no_insertion()
    print("\nAll predictive cooling tests passed.")
