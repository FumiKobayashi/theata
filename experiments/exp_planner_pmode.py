"""planner v3 (delivery トリガー整列) の動作イメージ検証 (2026-09-10)。

著者のイメージを命題化してイベントタイムライン上で計測する:
  P1 (PMode1): 「片側が LIZ へ納品中、反対側メモリが並び替えを行う」
      → 側 X の deliver 稼働区間 (PMode1 内) のうち、反対側 Y の sort が
        同時稼働している時間割合
  P2 (PMode2): 「測定中は左右同時に並び替えが動く」
      → Measure 区間のうち left-sort ∩ right-sort が同時稼働する時間割合

補助指標:
  - sort 時間の PMode1 / PMode2 への配分
  - 左右搬送の同時稼働率 (PMode 別)
  - 測定窓の全停止 (左右とも搬送ゼロ) 窓数 — 9/8 の v1 診断 (13 窓) との比較

op の出所は planner が emit 時に付ける origin タグ
(deliver / evict / sort / stage / flush) を使う。

Usage:
    python experiments/exp_planner_pmode.py            # d5, 1 round
    python experiments/exp_planner_pmode.py --rounds 3
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from ast_def import MeasureOp, ShuttleOp, SplitOp, MergeOp, SwapOp
from qccd_compiler import Scheduler, CompilerConfig
from event_scheduler import schedule_events
from exp_planner_v2 import run_variant  # 正準 fwd/rev + 配置持ち越し
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler

LEFT, RIGHT, LIZ = {0, 1, 2}, {4, 5, 6}, 3
TRANSPORT = (ShuttleOp, SplitOp, MergeOp, SwapOp)
zones_of = Scheduler()._zones_used


# ---------- 区間ユーティリティ ----------

def union(ivs):
    ivs = sorted(ivs)
    out = []
    for t0, t1 in ivs:
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1))
        else:
            out.append((t0, t1))
    return out


def isect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def compl(ivs, t0, t1):
    out, cur = [], t0
    for a, b in ivs:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < t1:
        out.append((cur, t1))
    return out


def total(ivs):
    return sum(t1 - t0 for t0, t1 in ivs)


# ---------- コンパイルとタイムライン ----------

def compile_events(d, cfg, rounds):
    """rounds 分の flat_ops をイベント時刻に割り当て、連結して返す。"""
    forward, reverse, _ = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)
    placement, zone_contents = initial, None
    intervals, offset = [], 0.0
    for r in range(1, rounds + 1):
        mod = __import__(forward if r % 2 == 1 else reverse)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=cfg).compile(dag, placement)
        ivs, offset = schedule_events(cr.flat_ops, t_offset=offset)
        intervals.extend(ivs)
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
    return intervals, offset


def side_of(op):
    z = zones_of(op)
    if z & LEFT:
        return 'L'
    if z & RIGHT:
        return 'R'
    return None


def analyze(intervals, t_end):
    meas = [(iv.t0, iv.t1) for iv in intervals if isinstance(iv.op, MeasureOp)]
    pmode2 = union(meas)
    pmode1 = compl(pmode2, 0.0, t_end)

    # side × category の稼働区間
    busy = {('L', 'transport'): [], ('R', 'transport'): [],
            ('L', 'deliver'): [], ('R', 'deliver'): [],
            ('L', 'evict'): [], ('R', 'evict'): [],
            ('L', 'sort'): [], ('R', 'sort'): []}
    for iv in intervals:
        if not isinstance(iv.op, TRANSPORT):
            continue
        s = side_of(iv.op)
        if s is None:
            continue
        seg = (iv.t0, iv.t1)
        busy[(s, 'transport')].append(seg)
        origin = getattr(iv.op, 'origin', None)
        if origin in ('deliver', 'evict', 'sort'):
            busy[(s, origin)].append(seg)
    busy = {k: union(v) for k, v in busy.items()}

    res = {'t_end_ms': t_end / 1000,
           'pmode1_ms': total(pmode1) / 1000,
           'pmode2_ms': total(pmode2) / 1000}

    # P1: PMode1 で「側 X の納品中に反対側 Y が sort」
    for x, y in (('L', 'R'), ('R', 'L')):
        dlv = isect(busy[(x, 'deliver')], pmode1)
        ov = isect(dlv, busy[(y, 'sort')])
        res[f'P1_{x}deliver_{y}sort_ms'] = total(ov) / 1000
        res[f'P1_{x}deliver_ms'] = total(dlv) / 1000
        res[f'P1_ratio_{x}{y}'] = (total(ov) / total(dlv)) if dlv else 0.0

    # P3 (9/10 著者観察): 「evict 中にも反対側が sort している」
    #   側 X の evict 稼働区間のうち、反対側 Y の sort が同時稼働する割合。
    #   同側 (X=Y) はゾーン競合で原理的に重ならないはずなので対照として併記
    for x, y in (('L', 'R'), ('R', 'L')):
        ev = busy[(x, 'evict')]
        res[f'P3_{x}evict_ms'] = total(ev) / 1000
        res[f'P3_{x}evict_{y}sort_ms'] = total(isect(ev, busy[(y, 'sort')])) / 1000
        res[f'P3_ratio_{x}{y}'] = (total(isect(ev, busy[(y, 'sort')])) / total(ev)
                                   if ev else 0.0)
        res[f'P3_sameside_{x}_ms'] = total(isect(ev, busy[(x, 'sort')])) / 1000

    # P2: PMode2 で左右 sort 同時稼働
    both_sort = isect(busy[('L', 'sort')], busy[('R', 'sort')])
    res['P2_both_sort_in_meas_ms'] = total(isect(both_sort, pmode2)) / 1000
    res['P2_ratio'] = (total(isect(both_sort, pmode2)) / total(pmode2)
                       if pmode2 else 0.0)

    # sort 時間の配分
    for s in ('L', 'R'):
        res[f'sort_{s}_pmode1_ms'] = total(isect(busy[(s, 'sort')], pmode1)) / 1000
        res[f'sort_{s}_pmode2_ms'] = total(isect(busy[(s, 'sort')], pmode2)) / 1000

    # 左右搬送の同時稼働率 (PMode 別)
    both_tr = isect(busy[('L', 'transport')], busy[('R', 'transport')])
    for name, region in (('pmode1', pmode1), ('pmode2', pmode2)):
        reg_t = total(region)
        res[f'lr_simul_{name}_ratio'] = (total(isect(both_tr, region)) / reg_t
                                         if reg_t else 0.0)

    # 測定窓の全停止 (左右とも搬送ゼロ) 窓数
    all_tr = union(busy[('L', 'transport')] + busy[('R', 'transport')])
    stalled = sum(1 for w in meas if not isect([w], all_tr))
    res['measure_windows'] = len(meas)
    res['all_stop_windows'] = stalled
    return res


def main():
    ap = argparse.ArgumentParser(description='PMode-level verification of v3')
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=1)
    args = ap.parse_args()

    variants = [
        ('v1', CompilerConfig()),
        ('v3 deliv h1', CompilerConfig(planner='v2', prefetch_depth=0,
                                       v2_sort_horizon=1,
                                       v2_sort_trigger='delivery')),
    ]
    results = {}
    for label, cfg in variants:
        intervals, t_end = compile_events(args.d, cfg, args.rounds)
        r = analyze(intervals, t_end)
        results[label] = r
        print(f"\n=== {label}  (d={args.d}, rounds={args.rounds}, "
              f"wall={r['t_end_ms']:.1f}ms | PMode1 {r['pmode1_ms']:.1f}ms / "
              f"PMode2 {r['pmode2_ms']:.1f}ms) ===")
        print(f"  P1 L-deliver 中に R-sort: {r['P1_Ldeliver_Rsort_ms']:.2f}ms"
              f" / L-deliver {r['P1_Ldeliver_ms']:.2f}ms"
              f" (ratio {r['P1_ratio_LR']:.1%})")
        print(f"  P1 R-deliver 中に L-sort: {r['P1_Rdeliver_Lsort_ms']:.2f}ms"
              f" / R-deliver {r['P1_Rdeliver_ms']:.2f}ms"
              f" (ratio {r['P1_ratio_RL']:.1%})")
        print(f"  P2 測定中の左右同時 sort: {r['P2_both_sort_in_meas_ms']:.2f}ms"
              f" / PMode2 {r['pmode2_ms']:.1f}ms (ratio {r['P2_ratio']:.1%})")
        print(f"  P3 L-evict 中に R-sort: {r['P3_Levict_Rsort_ms']:.2f}ms"
              f" / L-evict {r['P3_Levict_ms']:.2f}ms"
              f" (ratio {r['P3_ratio_LR']:.1%}, 同側対照 {r['P3_sameside_L_ms']:.2f}ms)")
        print(f"  P3 R-evict 中に L-sort: {r['P3_Revict_Lsort_ms']:.2f}ms"
              f" / R-evict {r['P3_Revict_ms']:.2f}ms"
              f" (ratio {r['P3_ratio_RL']:.1%}, 同側対照 {r['P3_sameside_R_ms']:.2f}ms)")
        print(f"  sort 配分: L pmode1 {r['sort_L_pmode1_ms']:.2f} / "
              f"pmode2 {r['sort_L_pmode2_ms']:.2f}ms, "
              f"R pmode1 {r['sort_R_pmode1_ms']:.2f} / "
              f"pmode2 {r['sort_R_pmode2_ms']:.2f}ms")
        print(f"  左右搬送同時稼働率: PMode1 {r['lr_simul_pmode1_ratio']:.1%}, "
              f"PMode2 {r['lr_simul_pmode2_ratio']:.1%}")
        print(f"  測定窓全停止: {r['all_stop_windows']} / {r['measure_windows']} 窓")

    out_dir = Path(__file__).parent.parent / 'results' / 'planner_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = out_dir / f'pmode_verify_{ts}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'d': args.d, 'rounds': args.rounds, 'results': results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
