"""QEC ラウンド境界でのフォノン引き継ぎ (phonon carry-over) のテスト。

背景:
  THeatA は既定でラウンドごとに新しい QCCDState を作るため、フォノン数と
  計画時近似時計がラウンド境界で 0 に戻る。これは加熱の効果を下限側に
  見積もる単純化 (論文 Appendix A)。
  QCCDCompiler.compile(initial_phonons=..., initial_clock_us=...) と
  exp_distance_scan の --carry-phonon は、この 0 リセットをやめて
  前ラウンド最終値を次ラウンドの初期値にするオプション。

検証項目:
- 既定 (オプションなし) では phonon_at_gates が変更前と完全一致する
- R=1 ではオプションの有無によらず一致する
- 引き継ぎありでは、ラウンド r+1 開始時の値がラウンド r の最終値に等しい
- 背景加熱を入れたとき、time_us と tr1 がラウンド境界で連続している

Usage:
    python tests/test_phonon_carryover.py
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent
for sub in ('src', 'circuits', 'experiments'):
    sys.path.insert(0, str(ROOT / sub))

import matplotlib                                   # noqa: E402
matplotlib.use('Agg')

from qccd_parser import parse                       # noqa: E402
from dag_builder import DAGBuilder                  # noqa: E402
from qccd_compiler import (                         # noqa: E402
    QCCDArchitecture, QCCDCompiler, QCCDState, CompilerConfig, MovementPlanner,
)
from sim_params import load_time_parameters         # noqa: E402

PARAMS_DIR = ROOT / 'params'
BASELINE = ROOT / 'tests' / 'data' / 'phonon_carryover_baseline.json'


# ------------------------------------------------------------------
# ユニット: QCCDState.seed_phonons
# ------------------------------------------------------------------

def test_seed_phonons_unit():
    """3 モードの注入、長さ 2 のフォールバック、未配置イオンの拒否"""
    arch = QCCDArchitecture.simple_mlm()
    state = QCCDState(architecture=arch)
    state.place_ion(0, 1)
    state.place_ion(1, 1)

    state.seed_phonons({0: (1.5, 2.5, 3.5), 1: (0.5, 0.25)})
    assert state.phonon_axial[0] == 1.5
    assert state.phonon_transversal[0] == 2.5
    assert state.phonon_transversal2[0] == 3.5
    # 長さ 2 → transversal2 は第 1 transverse モードと同値
    assert state.phonon_axial[1] == 0.5
    assert state.phonon_transversal[1] == 0.25
    assert state.phonon_transversal2[1] == 0.25

    for bad in ({7: (0.0, 0.0, 0.0)}, {0: (1.0,)}, {0: (1.0, 2.0, 3.0, 4.0)}):
        try:
            state.seed_phonons(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seed_phonons({bad!r}) should raise ValueError")
    print("test_seed_phonons_unit: OK")


def test_state_arg_rejects_seed_kwargs():
    """state 指定時に initial_phonons を渡したら黙って無視せずエラーにする"""
    arch = QCCDArchitecture.simple_mlm()
    planner = MovementPlanner(arch, CompilerConfig())
    state = QCCDState(architecture=arch)
    state.phonon_params = dict(planner.phonon_params)
    for ion in (0, 1, 2):
        state.place_ion(ion, 1)
    dag = DAGBuilder(parse(_CODE)).build()
    try:
        planner.plan(dag, state=state, initial_phonons={0: (1.0, 1.0, 1.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("plan(state=..., initial_phonons=...) should raise")
    print("test_state_arg_rejects_seed_kwargs: OK")


# ------------------------------------------------------------------
# compile() レベル: 小さい回路で線形性と引き継ぎ意味論を見る
# ------------------------------------------------------------------

_CODE = """
QUBIT 0 AT (1, 0)
QUBIT 1 AT (1, 1)
QUBIT 2 AT (1, 2)
MS 0 1 0.25
MS 1 2 0.25
MEASURE 2
"""
_PLACEMENT = {0: 1, 1: 1, 2: 1}


def _compile(tp=None, initial_phonons=None, initial_clock_us=None,
             placement=None):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(_CODE)).build()
    comp = QCCDCompiler(arch, config=CompilerConfig(), time_params=tp)
    return comp.compile(dag, dict(placement or _PLACEMENT),
                        initial_phonons=initial_phonons,
                        initial_clock_us=initial_clock_us)


def test_default_is_unchanged():
    """initial_phonons=None は「全モード 0 から」と完全に同じ結果"""
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    res_none = _compile(tp)
    zeros = {ion: (0.0, 0.0, 0.0) for ion in _PLACEMENT}
    res_zero = _compile(tp, initial_phonons=zeros, initial_clock_us=0.0)
    assert res_none.phonon_data['at_gates'] == res_zero.phonon_data['at_gates']
    assert (res_none.phonon_data['planning_clock_us']
            == res_zero.phonon_data['planning_clock_us'])
    print("test_default_is_unchanged: OK")


def test_uniform_seed_shifts_every_gate_by_the_same_amount():
    """一様な初期値 c は全ゲート記録を厳密に +c だけ平行移動させる。

    split / merge / shuttle / swap / advance_time はすべてアフィンで、
    merge の平均化は一様な定数を保つため、冷却なしなら厳密に成り立つ。
    """
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    base = _compile(tp)
    c_ax, c_tr1, c_tr2 = 0.7, 1.3, 2.1
    seeded = _compile(tp, initial_phonons={
        ion: (c_ax, c_tr1, c_tr2) for ion in _PLACEMENT})

    g0 = base.phonon_data['at_gates']
    g1 = seeded.phonon_data['at_gates']
    assert len(g0) == len(g1) and len(g0) > 0
    for a, b in zip(g0, g1):
        assert a['gate'] == b['gate']
        assert a['time_us'] == b['time_us']   # 時計はフォノン値に依存しない
        for ion, pa in a['phonons'].items():
            pb = b['phonons'][ion]
            assert abs((pb[0] - pa[0]) - c_ax) < 1e-9
            assert abs((pb[1] - pa[1]) - c_tr1) < 1e-9
            assert abs((pb[2] - pa[2]) - c_tr2) < 1e-9
    print("test_uniform_seed_shifts_every_gate_by_the_same_amount: OK")


def test_initial_clock_offsets_recorded_times():
    """initial_clock_us は記録時刻と planning_clock_us をそのままずらす"""
    tp = load_time_parameters()          # レート 0 → フォノンは時計に依存しない
    base = _compile(tp)
    offset = 12345.0
    shifted = _compile(tp, initial_clock_us=offset)

    assert (abs(shifted.phonon_data['planning_clock_us']
                - (base.phonon_data['planning_clock_us'] + offset)) < 1e-9)
    for a, b in zip(base.phonon_data['at_gates'],
                    shifted.phonon_data['at_gates']):
        assert abs((b['time_us'] - a['time_us']) - offset) < 1e-9
    print("test_initial_clock_offsets_recorded_times: OK")


class _CapturingPlanner(MovementPlanner):
    """plan 開始時の state を記録する planner (引き継ぎ値の検証用)"""

    def _make_initial_state(self, *args, **kwargs):
        state = super()._make_initial_state(*args, **kwargs)
        self.seeded_axial = dict(state.phonon_axial)
        self.seeded_transversal = dict(state.phonon_transversal)
        self.seeded_transversal2 = dict(state.phonon_transversal2)
        self.seeded_clock_us = state.clock_us
        return state


def test_next_round_starts_at_previous_round_end():
    """ラウンド r+1 の開始値 == ラウンド r の最終値 (3 モード + 時計)"""
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    arch = QCCDArchitecture.simple_mlm()

    r1 = _compile(tp)
    pd1 = r1.phonon_data
    seed = {ion: (pd1['per_ion_axial'][ion],
                  pd1['per_ion_transversal'][ion],
                  pd1['per_ion_transversal2'][ion])
            for ion in pd1['per_ion_axial']}

    planner = _CapturingPlanner(arch, CompilerConfig(), time_params=tp)
    dag = DAGBuilder(parse(_CODE)).build()
    planner.plan(dag, dict(_PLACEMENT), initial_phonons=seed,
                 initial_clock_us=pd1['planning_clock_us'])

    assert planner.seeded_axial == pd1['per_ion_axial']
    assert planner.seeded_transversal == pd1['per_ion_transversal']
    assert planner.seeded_transversal2 == pd1['per_ion_transversal2']
    assert planner.seeded_clock_us == pd1['planning_clock_us']
    # 引き継がない場合は 0 から始まる (対照)
    planner0 = _CapturingPlanner(arch, CompilerConfig(), time_params=tp)
    planner0.plan(DAGBuilder(parse(_CODE)).build(), dict(_PLACEMENT))
    assert set(planner0.seeded_transversal.values()) == {0.0}
    assert planner0.seeded_clock_us == 0.0
    print("test_next_round_starts_at_previous_round_end: OK")


# ------------------------------------------------------------------
# exp_distance_scan レベル: マルチラウンドの実際の経路
# ------------------------------------------------------------------

def _rounds_at_gates(results):
    return [(r.phonon_data or {}).get('at_gates', []) for r in results]


def _build(rounds, carry_phonon):
    from exp_distance_scan import build_noisy_circuit, DISTANCE_CIRCUITS
    fwd, rev, obs = DISTANCE_CIRCUITS[3]
    _, results = build_noisy_circuit(fwd, rev, rounds, obs, warn_on_clip=False,
                                     carry_phonon=carry_phonon)
    return results


def test_default_matches_stored_baseline():
    """carry_phonon=False の d=3, R=3 が変更前の基準データと完全一致する"""
    reference = json.loads(BASELINE.read_text(encoding='utf-8'))
    results = _build(reference['rounds'], carry_phonon=False)

    got = []
    for per_round in _rounds_at_gates(results):
        got.extend(per_round)
    # JSON ラウンドトリップで phonons の int キーが文字列になるので揃える
    got_json = json.loads(json.dumps(got, sort_keys=True))
    assert got_json == reference['at_gates'], (
        "carry_phonon=False の phonon_at_gates が基準データと一致しない "
        "(既定動作が変わっている)")
    assert ([(r.phonon_data or {}).get('planning_clock_us') for r in results]
            == reference['planning_clock_us'])
    print(f"test_default_matches_stored_baseline: OK ({len(got)} gates)")


def test_single_round_identical_with_and_without_option():
    """R=1 では引き継ぎの有無で結果が一致する (引き継ぐ前ラウンドがない)"""
    off = _rounds_at_gates(_build(1, carry_phonon=False))
    on = _rounds_at_gates(_build(1, carry_phonon=True))
    assert off == on
    print("test_single_round_identical_with_and_without_option: OK")


def test_carry_over_is_continuous_across_round_boundaries():
    """引き継ぎあり: 時計と tr1 がラウンド境界で途切れない"""
    rounds = 3
    off = _build(rounds, carry_phonon=False)
    on = _build(rounds, carry_phonon=True)

    # ラウンド 1 は両者同一 (引き継ぐ前ラウンドがない)
    assert (off[0].phonon_data['at_gates']
            == on[0].phonon_data['at_gates'])

    for r in range(1, rounds):
        prev, cur = on[r - 1].phonon_data, on[r].phonon_data
        # 時計の連続性: 当ラウンド最初のゲートは前ラウンド終端以降に起きる
        assert cur['at_gates'][0]['time_us'] >= prev['planning_clock_us'] - 1e-9
        assert cur['planning_clock_us'] > prev['planning_clock_us']
        # リセットありなら逆に 0 付近から再出発する (対照)
        off_cur = off[r].phonon_data
        assert (off_cur['at_gates'][0]['time_us']
                < off[r - 1].phonon_data['planning_clock_us'])

        # tr1 の連続性: 加熱と merge 平均は値を [min, max] の外へ出さないので、
        # 当ラウンド最初のゲートの tr1 は前ラウンド終端の最小値を下回らない
        prev_min_tr1 = min(prev['per_ion_transversal'].values())
        first_gate_tr1 = [p[1] for p in cur['at_gates'][0]['phonons'].values()]
        assert min(first_gate_tr1) >= prev_min_tr1 - 1e-9
        # かつ、リセットありの同じゲートより必ず熱い
        off_first = [p[1] for p in off_cur['at_gates'][0]['phonons'].values()]
        assert min(first_gate_tr1) > min(off_first)
    print("test_carry_over_is_continuous_across_round_boundaries: OK")


def test_carry_over_grows_with_rounds():
    """引き継ぎありでは最終 tr1 がラウンド数とともに単調に増える"""
    finals = {}
    for rounds in (1, 2, 4):
        results = _build(rounds, carry_phonon=True)
        finals[rounds] = max(
            results[-1].phonon_data['per_ion_transversal'].values())
    assert finals[1] < finals[2] < finals[4], finals
    print(f"test_carry_over_grows_with_rounds: OK (max tr1 {finals})")


if __name__ == "__main__":
    test_seed_phonons_unit()
    test_state_arg_rejects_seed_kwargs()
    test_default_is_unchanged()
    test_uniform_seed_shifts_every_gate_by_the_same_amount()
    test_initial_clock_offsets_recorded_times()
    test_next_round_starts_at_previous_round_end()
    test_default_matches_stored_baseline()
    test_single_round_identical_with_and_without_option()
    test_carry_over_is_continuous_across_round_boundaries()
    test_carry_over_grows_with_rounds()
    print("\nAll phonon carry-over tests passed.")
