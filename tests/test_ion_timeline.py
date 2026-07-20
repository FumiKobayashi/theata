"""ion_timeline (per-ion イベントタイムライン) のテスト。

Usage:
    python tests/test_ion_timeline.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ast_def import GateOp, ResetOp, MeasureOp, SplitOp, TimeStep
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler
from parallel_estimator import ParallelEstimator
from estimator import TimeParameters
from per_liz_compiler import Segment, PerLIZResult
from ion_timeline import (
    op_intervals_from_timesteps, op_intervals_from_streams,
    build_ion_events, validate_order_against, compute_ion_gaps,
    gaps_from_compile_results, is_anchor_op,
)

T_R = 4.0
T_MS = 59.0
T_SPLIT = 100.0
T_RESET = 41.0
T_MEAS = 1000.0


def _ms(a, b):
    return GateOp(type='MS', targets=[a, b], params=[0.25], zone_id=3)


def _r(q):
    return GateOp(type='R', targets=[q], params=[1.57, 0.0], zone_id=3)


def test_intervals_and_gaps_basic():
    """手組み timestep: transport が gap に、gate が busy になる"""
    # ts0: Split(dur 100) と MS(2,3)(dur 59) が並列 → timestep 長 100
    # ts1: MS(0,1) → t=100 から
    ts = [
        TimeStep(ops=[SplitOp(ion_id=7, dst_zone_id=2, src_zone_id=1), _ms(2, 3)]),
        TimeStep(ops=[_ms(0, 1)]),
    ]
    intervals, t_end = op_intervals_from_timesteps(ts)
    assert t_end == T_SPLIT + T_MS, t_end

    events = build_ion_events(intervals)
    # Split はイオンイベントを作らない (ion 7 は登場しない)
    assert 7 not in events
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=t_end)

    ms01 = ts[1].ops[0]
    ms23 = ts[0].ops[1]
    # ion 0/1: t=0..100 が idle → MS(0,1) の直前 gap = 100
    assert gaps.gaps_by_op[id(ms01)] == {0: T_SPLIT, 1: T_SPLIT}
    # ion 2/3: t=0 で即ゲート → gap 0 (エントリなし)
    assert id(ms23) not in gaps.gaps_by_op
    # trailing: ion 2/3 は 59 で終了 → 100 残り (Split 分 41 + ts1 分 59)
    assert abs(gaps.trailing_idle[2] - (t_end - T_MS)) < 1e-9
    assert abs(gaps.trailing_idle[0] - 0.0) < 1e-9
    print("test_intervals_and_gaps_basic: OK")


def test_reset_absorption_and_r_carryover():
    """RESET はギャップを吸収、非 anchor の R gate はギャップを持ち越す"""
    ts = [
        TimeStep(ops=[SplitOp(ion_id=7, dst_zone_id=2, src_zone_id=1)]),  # 100
        TimeStep(ops=[ResetOp(targets=[9], zone_id=3)]),                   # 41
        TimeStep(ops=[SplitOp(ion_id=8, dst_zone_id=2, src_zone_id=1)]),  # 100
        TimeStep(ops=[_r(9)]),                                             # 4
        TimeStep(ops=[SplitOp(ion_id=8, dst_zone_id=1, src_zone_id=2)]),  # 100
        TimeStep(ops=[_ms(0, 9)]),                                         # 59
        TimeStep(ops=[MeasureOp(targets=[9], zone_id=3)]),                 # 1000
    ]
    intervals, t_end = op_intervals_from_timesteps(ts)
    events = build_ion_events(intervals)
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=t_end)

    ms = ts[5].ops[0]
    meas = ts[6].ops[0]
    # ion 9: RESET が t=100 までの idle を吸収。
    #   RESET 後: split(100) idle + R(busy 4) + split(100) idle → MS 直前 gap = 200
    assert abs(gaps.gaps_by_op[id(ms)][9] - 200.0) < 1e-9
    # ion 0: 先頭から MS まで全部 idle = 100+41+100+4+100 = 345
    assert abs(gaps.gaps_by_op[id(ms)][0] - 345.0) < 1e-9
    # MEASURE 直前の ion 9: MS 終了 (t=404) 直後に測定 → gap 0
    assert 9 not in gaps.gaps_by_op.get(id(meas), {})
    # 保存則 (ResetOp を持たない ion 0): gaps + busy + trailing = t_end
    total_busy_0 = T_MS
    total_gap_0 = sum(g.get(0, 0.0) for g in gaps.gaps_by_op.values())
    assert abs(total_gap_0 + total_busy_0 + gaps.trailing_idle[0] - t_end) < 1e-9
    print("test_reset_absorption_and_r_carryover: OK")


def test_multi_liz_streams_and_order_validation():
    """multi-LIZ: 共通時計での cross-core イオン混合 + 順序/非重複 assert"""
    arch = QCCDArchitecture.simple_mlm()
    ms_a = _ms(0, 5)   # core 3 が ion 5 を先に使う
    ms_b = _ms(1, 5)   # core 7 が後で使う (sync 後)

    def _seg(core, op, start):
        s = Segment(core_id=core, label='row0', ops=[op])
        s.timesteps = [TimeStep(ops=[op])]
        s.start_us = start
        s.end_us = start + T_MS
        s.duration_us = T_MS
        return s

    result = PerLIZResult(
        segments=[_seg(3, ms_a, 0.0), _seg(7, ms_b, T_MS)],
        streams={3: [0], 7: [1]}, syncs=[],
        merged_flat=[ms_a, ms_b],
        initial_placement={}, state=None, arch=arch,
        wall_clock_us=2 * T_MS,
    )
    intervals, t_end = op_intervals_from_streams(result)
    events = build_ion_events(intervals)
    validate_order_against(events, result.merged_flat)   # 正順 → 通る
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=t_end)
    # ion 5 は連続稼働 → ms_b 直前の gap なし
    assert 5 not in gaps.gaps_by_op.get(id(ms_b), {})
    # ion 1 は t=0..59 idle → gap 59
    assert abs(gaps.gaps_by_op[id(ms_b)][1] - T_MS) < 1e-9

    # 併合順が物理順と食い違う → assert が検出 (cross-LIZ 順序テスト)
    try:
        validate_order_against(events, [ms_b, ms_a])
        assert False, "order violation should raise"
    except AssertionError as e:
        assert "食い違" in str(e)

    # 同一イオンの busy 区間が重なる → assert が検出 (race 検出)
    result_bad = PerLIZResult(
        segments=[_seg(3, ms_a, 0.0), _seg(7, ms_b, 30.0)],
        streams={3: [0], 7: [1]}, syncs=[],
        merged_flat=[ms_a, ms_b],
        initial_placement={}, state=None, arch=arch,
        wall_clock_us=30.0 + T_MS,
    )
    intervals_bad, _ = op_intervals_from_streams(result_bad)
    try:
        build_ion_events(intervals_bad)
        assert False, "overlap should raise"
    except AssertionError as e:
        assert "重複" in str(e)
    print("test_multi_liz_streams_and_order_validation: OK")


def test_compile_integration_and_multi_round():
    """実コンパイル結果で: t_end = estimator 合計、保存則、ラウンド連結"""
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

    def compile_once():
        dag = DAGBuilder(parse(code)).build()
        return QCCDCompiler(arch).compile(dag, dict(initial))

    r1, r2 = compile_once(), compile_once()
    anchor_ops, gaps = gaps_from_compile_results([r1, r2])

    # t_end = 両ラウンドの並列時間合計 (max 一律オフセット)
    pe = ParallelEstimator()
    expected = (pe.estimate(r1.timesteps).total_time
                + pe.estimate(r2.timesteps).total_time)
    assert abs(gaps.t_end - expected) < 1e-6

    # anchor 数 = RESET/MS/MEASURE の総数 × 2 ラウンド
    assert len(anchor_ops) == 2 * 4
    assert all(is_anchor_op(op) for op in anchor_ops)

    # 保存則 (data ion 0: ResetOp なし): gaps + busy + trailing = t_end
    busy_0 = 0.0
    for cr in (r1, r2):
        for op in cr.flat_ops:
            if isinstance(op, GateOp) and 0 in op.targets:
                busy_0 += T_MS if op.type == 'MS' else T_R
    gap_0 = sum(g.get(0, 0.0) for g in gaps.gaps_by_op.values())
    assert abs(gap_0 + busy_0 + gaps.trailing_idle[0] - gaps.t_end) < 1e-6

    # ラウンド 2 の最初の anchor に、ラウンド 1 終端からの繰り越しが載る:
    # ion 0 のラウンド 2 最初のギャップ >= ラウンド 1 単体の trailing
    _, gaps_r1 = gaps_from_compile_results([r1])
    r2_anchor_ids = {id(op) for op in r2.flat_ops if is_anchor_op(op)}
    r2_first_gap_0 = None
    for op in anchor_ops:
        if id(op) in r2_anchor_ids and 0 in gaps.gaps_by_op.get(id(op), {}):
            r2_first_gap_0 = gaps.gaps_by_op[id(op)][0]
            break
    assert r2_first_gap_0 is not None
    assert r2_first_gap_0 >= gaps_r1.trailing_idle[0] - 1e-9
    print("test_compile_integration_and_multi_round: OK")


if __name__ == "__main__":
    test_intervals_and_gaps_basic()
    test_reset_absorption_and_r_carryover()
    test_multi_liz_streams_and_order_validation()
    test_compile_integration_and_multi_round()
    print("\nAll ion_timeline tests passed.")
