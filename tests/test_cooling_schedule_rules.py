"""cooling を広義に見たときのスケジュール排他性の検証。

ルール:
  A (同一 timestep): CoolOp またはエピソード構成 op (coolant の
    split/shuttle/merge 往復) を含む timestep に Measure/Gate/Reset が
    同居しない。
  B (エピソード時間スパン): 各 merge 冷却エピソードの [開始 ts, 終了 ts]
    の間のどの timestep にも Measure/Gate/Reset が現れない
    (= 「coolant 合流の最中に測定が開始」がゾーンが違っても起きていない)。
    スパン内の輸送 op (memory 整列等) は許容。

Usage:
    python tests/test_cooling_schedule_rules.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ast_def import CoolOp, MeasureOp, ResetOp, GateOp
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig

BUSY_KINDS = (GateOp, MeasureOp, ResetOp)


def check_cooling_schedule_rules(timesteps, episodes):
    """CompileResult.timesteps と planner.episodes からルール A/B を検証する。

    Returns:
        dict: {'same_ts': [(ts, [op種別,...]), ...],
               'span': [(episode番号, ts, op種別), ...],
               'span_coresidents': {op種別: 件数} (違反でない同居 op の内訳),
               'n_episodes': int}
    """
    op_ts = {}
    for t, ts in enumerate(timesteps):
        for op in ts.ops:
            op_ts[id(op)] = t
    episode_ids = set()
    for ep in episodes:
        episode_ids.update(ep)

    same_ts = []
    for t, ts in enumerate(timesteps):
        has_cool = any(isinstance(op, CoolOp) or id(op) in episode_ids
                       for op in ts.ops)
        if not has_cool:
            continue
        bad = [type(op).__name__ for op in ts.ops
               if isinstance(op, BUSY_KINDS)]
        if bad:
            same_ts.append((t, bad))

    span = []
    span_coresidents = {}
    for k, ep in enumerate(episodes):
        tss = sorted(op_ts[i] for i in ep if i in op_ts)
        if not tss:
            continue
        lo, hi = tss[0], tss[-1]
        for t in range(lo, hi + 1):
            for op in timesteps[t].ops:
                if isinstance(op, BUSY_KINDS):
                    span.append((k, t, type(op).__name__))
                elif id(op) not in episode_ids:
                    name = type(op).__name__
                    span_coresidents[name] = span_coresidents.get(name, 0) + 1
    return {'same_ts': same_ts, 'span': span,
            'span_coresidents': span_coresidents,
            'n_episodes': len(episodes)}


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


def _check(cfg):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(CODE)).build()
    comp = QCCDCompiler(arch, config=cfg)
    res = comp.compile(dag, dict(INITIAL))
    rep = check_cooling_schedule_rules(res.timesteps, comp._planner.episodes)
    return rep


def test_merge_cool_rules():
    rep = _check(CompilerConfig(
        merge_cool_threshold_transversal=0.05,
        cool_ancilla_after_measure_us=2000.0))
    assert rep['n_episodes'] >= 1, "エピソードが発動しているはず"
    assert rep['same_ts'] == [], f"Rule A violation: {rep['same_ts']}"
    assert rep['span'] == [], f"Rule B violation: {rep['span']}"
    print(f"test_merge_cool_rules: OK ({rep['n_episodes']} episodes, "
          f"span coresidents={rep['span_coresidents']})")


def test_predictive_cool_rules():
    rep = _check(CompilerConfig(
        predictive_cool_threshold_transversal=0.0,
        predictive_cool_max_iters=3,
        cool_ancilla_after_measure_us=2000.0))
    assert rep['n_episodes'] >= 2, "複数エピソードが発動しているはず"
    assert rep['same_ts'] == [], f"Rule A violation: {rep['same_ts']}"
    assert rep['span'] == [], f"Rule B violation: {rep['span']}"
    print(f"test_predictive_cool_rules: OK ({rep['n_episodes']} episodes, "
          f"span coresidents={rep['span_coresidents']})")


if __name__ == "__main__":
    test_merge_cool_rules()
    test_predictive_cool_rules()
    print("\nAll cooling schedule rule tests passed.")
