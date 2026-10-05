"""Background heating rate (時間比例連続加算) のテスト。

モデル (2026-09-23 先生指示 + 10/2 会議):
- 経過時間比例の加熱を 3 モード独立のレート (quanta/s) で連続加算
- 時計 = 計画時近似時計 (emit した op の duration の直列和, QCCDState.clock_us)
- レート 0 (既定) では時計だけ進み、フォノンは既存挙動と完全一致

Usage:
    python tests/test_heating_rate.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import (
    QCCDArchitecture, QCCDCompiler, QCCDState, CompilerConfig,
)
from estimator import TimeParameters
from sim_params import load_time_parameters, phonon_params_from

PARAMS_DIR = Path(__file__).parent.parent / 'params'


def test_advance_time_unit():
    """advance_time: 時計の加算 + 全イオンへの一様加熱 (tr2 フォールバック込み)"""
    arch = QCCDArchitecture.simple_mlm()
    state = QCCDState(architecture=arch)
    state.phonon_params = phonon_params_from(TimeParameters(
        heating_rate_axial_per_s=5.5,
        heating_rate_transversal_per_s=4.8,
        heating_rate_transversal2_per_s=3.6,
    ))
    state.place_ion(0, 1)
    state.place_ion_any(1, 5)   # 別ゾーンのイオンも一様に加熱される

    state.advance_time(500_000.0)   # 0.5 s
    assert state.clock_us == 500_000.0
    for ion in (0, 1):
        assert abs(state.phonon_axial[ion] - 2.75) < 1e-9
        assert abs(state.phonon_transversal[ion] - 2.4) < 1e-9
        assert abs(state.phonon_transversal2[ion] - 1.8) < 1e-9

    # tr2 未指定 → 第 1 モードのレートにフォールバック
    state2 = QCCDState(architecture=arch)
    state2.phonon_params = phonon_params_from(TimeParameters(
        heating_rate_transversal_per_s=4.8))
    state2.place_ion(0, 1)
    state2.advance_time(1_000_000.0)
    assert abs(state2.phonon_transversal2[0] - 4.8) < 1e-9
    assert state2.phonon_axial[0] == 0.0    # axial レート未指定 = 0

    # レート 0: 時計だけ進む
    state3 = QCCDState(architecture=arch)
    state3.phonon_params = phonon_params_from(TimeParameters())
    state3.place_ion(0, 1)
    state3.advance_time(1000.0)
    assert state3.clock_us == 1000.0
    assert state3.phonon_axial[0] == 0.0
    print("test_advance_time_unit: OK")


_CODE = """
QUBIT 0 AT (1, 0)
QUBIT 1 AT (1, 1)
QUBIT 2 AT (1, 2)
MS 0 1 0.25
MS 1 2 0.25
MEASURE 2
"""


def _compile(tp):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(_CODE)).build()
    comp = QCCDCompiler(arch, config=CompilerConfig(), time_params=tp)
    return comp.compile(dag, {0: 1, 1: 1, 2: 1})


def test_zero_rate_backward_compat():
    """レート 0 (既定): フォノン値が従来と完全一致し、時計は記録される"""
    res = _compile(load_time_parameters())
    pd = res.phonon_data
    assert pd['planning_clock_us'] > 0.0
    # 全ゲート記録に time_us が付き、単調非減少
    times = [g['time_us'] for g in pd['at_gates']]
    assert times == sorted(times)
    assert times[-1] <= pd['planning_clock_us']
    # レート 0 では per-op コストのみ → 小数点以下が per-op 値の組合せに
    # なることまでは見ず、「heating 項が入っていない」ことを Mainz 版との
    # 差分で確認する (test_heating_changes_phonons)
    print("test_zero_rate_backward_compat: OK")


def test_heating_changes_phonons():
    """Mainz レート: 同一回路でフォノンが単調に増え、増分がレート×時計と整合"""
    res0 = _compile(load_time_parameters())
    res1 = _compile(load_time_parameters(PARAMS_DIR / 'mainz_heating.json'))
    pd0, pd1 = res0.phonon_data, res1.phonon_data

    # op 列・時計は同一 (heating は計画判断を変えない / 閾値ポリシー無効時)
    assert pd0['planning_clock_us'] == pd1['planning_clock_us']
    assert len(res0.flat_ops) == len(res1.flat_ops)

    # 最終 per-ion 値の増分 = レート × そのイオンの経過時間 (≤ 時計終端)。
    # merge 平均化を挟むので厳密比較はせず、上界と正の増分のみ検査。
    t_s = pd1['planning_clock_us'] * 1e-6
    for ion, v0 in pd0['per_ion_transversal'].items():
        v1 = pd1['per_ion_transversal'][ion]
        assert v1 > v0
        assert v1 - v0 <= 4.8 * t_s + 1e-9
    for ion, v0 in pd0['per_ion_transversal2'].items():
        v1 = pd1['per_ion_transversal2'][ion]
        assert v0 < v1 and v1 - v0 <= 3.6 * t_s + 1e-9
    for ion, v0 in pd0['per_ion_axial'].items():
        v1 = pd1['per_ion_axial'][ion]
        assert v0 < v1 and v1 - v0 <= 5.5 * t_s + 1e-9

    # ゲート時記録にも heating が乗る (tr1 比較)
    for g0, g1 in zip(pd0['at_gates'], pd1['at_gates']):
        assert g0['gate'] == g1['gate']
        for ion, p0 in g0['phonons'].items():
            p1 = g1['phonons'][ion]
            if g1['time_us'] > 0:
                assert p1[1] >= p0[1]
    print("test_heating_changes_phonons: OK")


def test_heating_interacts_with_threshold_cooling():
    """閾値冷却が heating 込みの値で発動する (レートを極端にして確認)"""
    tp = load_time_parameters(overrides={
        # 1 ms で transversal +100 → 最初の移動だけで閾値 1.0 を超える
        'heating_rate_transversal_per_s': 100_000.0,
    })
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(_CODE)).build()
    comp = QCCDCompiler(
        arch,
        config=CompilerConfig(cool_threshold_transversal=1.0),
        time_params=tp)
    res = comp.compile(dag, {0: 1, 1: 1, 2: 1})
    from ast_def import CoolOp
    n_cool = sum(1 for op in res.flat_ops if isinstance(op, CoolOp))
    assert n_cool >= 1, "heating 由来の閾値超過で CoolOp が入るはず"

    # 同条件でレート 0 なら CoolOp は入らない (per-op コストでは閾値未達)
    comp0 = QCCDCompiler(
        arch,
        config=CompilerConfig(cool_threshold_transversal=1.0),
        time_params=load_time_parameters())
    res0 = comp0.compile(DAGBuilder(parse(_CODE)).build(), {0: 1, 1: 1, 2: 1})
    n_cool0 = sum(1 for op in res0.flat_ops if isinstance(op, CoolOp))
    assert n_cool0 == 0
    print("test_heating_interacts_with_threshold_cooling: OK")


if __name__ == "__main__":
    test_advance_time_unit()
    test_zero_rate_backward_compat()
    test_heating_changes_phonons()
    test_heating_interacts_with_threshold_cooling()
    print("\nAll heating rate tests passed.")
