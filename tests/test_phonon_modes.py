"""フォノン 3 モード化 (axial + transverse 2 本) のテスト。

2026-10-02 会議決定: 1 イオンあたり 3 パラメータを追跡する。
- phonon_transversal  = 第 1 transverse モード (誤り参照モード)
- phonon_transversal2 = 第 2 transverse モード
- *_transversal2 パラメータ未指定時は第 1 モードと同コスト (フォールバック)

Usage:
    python tests/test_phonon_modes.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qccd_compiler import QCCDArchitecture, QCCDState
from estimator import TimeParameters
from sim_params import load_time_parameters, phonon_params_from


def test_params_fallback():
    """*_transversal2 = None なら第 1 モードと同値、指定時はその値"""
    ph = phonon_params_from()
    assert ph['split_transversal2'] == ph['split_transversal']
    assert ph['merge_transversal2'] == ph['merge_transversal']
    assert ph['swap_transversal2'] == ph['swap_transversal']
    assert ph['shuttle_transversal2'] == ph['shuttle_transversal']
    assert ph['cool_residual_transversal2'] == ph['cool_residual_transversal']

    tp = load_time_parameters(overrides={
        'phonon_merge_transversal2': 0.03,
        'phonon_cool_residual_transversal2': 0.2,
    })
    ph2 = phonon_params_from(tp)
    assert ph2['merge_transversal2'] == 0.03
    assert ph2['merge_transversal'] == 0.05   # 第 1 モードは不変
    assert ph2['cool_residual_transversal2'] == 0.2
    print("test_params_fallback: OK")


def _fresh_state(**tp_kwargs) -> QCCDState:
    arch = QCCDArchitecture.simple_mlm()
    state = QCCDState(architecture=arch)
    state.phonon_params = phonon_params_from(TimeParameters(**tp_kwargs))
    return state


def test_op_costs_update_mode2():
    """swap / split / shuttle が第 2 モードにも加算される"""
    state = _fresh_state(phonon_swap_transversal2=0.2,
                         phonon_split_transversal2=0.01,
                         phonon_shuttle_transversal2=0.08)
    for ion in (0, 1):
        state.place_ion(ion, 1)   # Memory zone 1

    state.swap_ions(0, 1)         # 配列は [1, 0] に
    assert state.phonon_transversal[0] == 0.1    # 第 1 モード: 既定 0.1
    assert state.phonon_transversal2[0] == 0.2   # 第 2 モード: 指定値
    assert state.phonon_transversal2[1] == 0.2

    state.split_chain(1, 0)       # 左端の ion 1 だけを左 Temp (zone 0) へ
    assert abs(state.phonon_transversal2[1] - (0.2 + 0.01)) < 1e-12
    assert state.phonon_transversal2[0] == 0.2   # 残留側は不変

    state.apply_shuttle_phonon(1, dst_chain_length=0)
    assert abs(state.phonon_transversal2[1] - (0.21 + 0.08)) < 1e-12
    print("test_op_costs_update_mode2: OK")


def test_merge_weighted_average_mode2():
    """merge = 重み付き平均 (n_a*s + n_b*t)/(s+t) + コスト e が 3 モード全てで成立"""
    state = _fresh_state()
    # chain A: zone 1 (Memory) に 3 イオン、chain B: zone 2 (Temp) に 1 イオン
    for ion in (0, 1, 2):
        state.place_ion(ion, 1)
    state.place_ion_any(3, 2)
    for ion in (0, 1, 2):
        state.phonon_axial[ion] = 12.0
        state.phonon_transversal[ion] = 4.0
        state.phonon_transversal2[ion] = 2.0
    state.phonon_axial[3] = 0.0
    state.phonon_transversal[3] = 0.0
    state.phonon_transversal2[3] = 0.0

    state.merge_chains(1, 2)
    # (4.0*3 + 0.0*1)/4 = 3.0 (+ merge_transversal 0.05)
    for ion in (0, 1, 2, 3):
        assert abs(state.phonon_axial[ion] - (9.0 + 10.0)) < 1e-12
        assert abs(state.phonon_transversal[ion] - (3.0 + 0.05)) < 1e-12
        assert abs(state.phonon_transversal2[ion] - (1.5 + 0.05)) < 1e-12
    print("test_merge_weighted_average_mode2: OK")


def test_cool_residual_mode2():
    """cool_chain が第 2 モードも残留値まで下げる (下回っていればそのまま)"""
    state = _fresh_state(phonon_cool_residual_transversal=0.1,
                         phonon_cool_residual_transversal2=0.3)
    state.place_ion(0, 1)
    state.place_ion(1, 1)
    state.phonon_transversal[0] = 1.0
    state.phonon_transversal2[0] = 5.0
    state.phonon_transversal2[1] = 0.05   # 残留値より低い → そのまま

    state.cool_chain(1)
    assert state.phonon_transversal[0] == 0.1
    assert state.phonon_transversal2[0] == 0.3
    assert state.phonon_transversal2[1] == 0.05
    print("test_cool_residual_mode2: OK")


def test_record_gate_phonon_triplet():
    """ゲート時記録が (axial, tr1, tr2) の 3 つ組になる (p[0]/p[1] は後方互換)"""
    state = _fresh_state()
    state.place_ion(0, 1)
    state.phonon_axial[0] = 1.0
    state.phonon_transversal[0] = 2.0
    state.phonon_transversal2[0] = 3.0

    state.record_gate_phonon('MS', [0])
    rec = state.phonon_at_gates[-1]['phonons'][0]
    assert len(rec) == 3
    assert rec[0] == 1.0 and rec[1] == 2.0 and rec[2] == 3.0
    print("test_record_gate_phonon_triplet: OK")


if __name__ == "__main__":
    test_params_fallback()
    test_op_costs_update_mode2()
    test_merge_weighted_average_mode2()
    test_cool_residual_mode2()
    test_record_gate_phonon_triplet()
    print("\nAll phonon 3-mode tests passed.")
