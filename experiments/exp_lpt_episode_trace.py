"""LPT episode trace: cycles per episode along the schedule.

2026-11-05 meeting follow-up (clarified request): when the LIZ
predictive trigger fires, how many averaging cycles does the episode
run, and how does that change between the early and the late part of
the schedule? This script wraps MovementPlanner._maybe_liz_pt and
records, for every MS gate, the schedule position (MS index and round),
whether an episode fired, its cycle count, its exit reason
(threshold reached / floor break / N_max / order break), and the
entry and exit phonon values of the gate pair.

Conditions match fig4 (t = 1 ms, n_res = 0, Mainz heating, LPT temp
N_max = 32), swept over theta = 2.8 / 1.5 / 0.2 and d = 3 / 5 / 7.
Compile only.

Output: results/lpt_episode_trace_1105/trace.json
Usage:  python experiments/exp_lpt_episode_trace.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import exp_cooling_distance_scan as E  # noqa: E402 (src/circuits paths)
import qccd_compiler as QC  # noqa: E402
from sim_params import load_time_parameters  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "lpt_episode_trace_1105"

# 共有カウンタをクラス属性に (hasattr True でインスタンス初期化がスキップ
# される — diag_lpt_floor 11/3 と同じ仕掛け)。エピソードの cycles / 終了
# 理由はこのカウンタの呼び出し前後差分から取る。
SHARED = Counter()
QC.MovementPlanner.lpt_skips = SHARED

TRACE = []          # 1 MS ゲート = 1 レコード
STATE = {"ms_idx": 0, "round_of": {}, "round": 0}
_orig = QC.MovementPlanner._maybe_liz_pt


def _traced(self, op, state):
    is_ms = isinstance(op, QC.GateOp) and op.type == "MS" and op.targets
    if not is_ms:
        return _orig(self, op, state)
    # 新しい planner インスタンス = 新しいラウンドのコンパイル
    if id(self) not in STATE["round_of"]:
        STATE["round"] += 1
        STATE["round_of"][id(self)] = STATE["round"]
    before = dict(SHARED)
    entry = max(state.phonon_transversal.get(t, 0.0) for t in op.targets)
    out = _orig(self, op, state)
    after = SHARED
    fired = after["fired"] > before.get("fired", 0)
    rec = {"ms": STATE["ms_idx"], "round": STATE["round_of"][id(self)],
           "fired": fired, "entry": round(entry, 3)}
    if fired:
        cycles = next((int(k[7:]) for k in after
                       if k.startswith("cycles_")
                       and after[k] > before.get(k, 0)), None)
        if after["floor_break"] > before.get("floor_break", 0):
            reason = "floor"
        elif after["order_break"] > before.get("order_break", 0):
            reason = "order"
        elif cycles is not None and cycles >= 32:
            reason = "nmax"
        else:
            reason = "threshold"
        exit_v = max(state.phonon_transversal.get(t, 0.0)
                     for t in op.targets)
        rec.update({"cycles": cycles, "reason": reason,
                    "exit": round(exit_v, 3)})
    TRACE.append(rec)
    STATE["ms_idx"] += 1
    return out


QC.MovementPlanner._maybe_liz_pt = _traced


def run(d, theta):
    SHARED.clear()
    TRACE.clear()
    STATE.update({"ms_idx": 0, "round_of": {}, "round": 0})
    tp = load_time_parameters(str(ROOT / "params" / "mainz_heating.json"),
                              overrides={"cool": 1000.0})
    cfg = dict(resident_coolant_ids=[9000, 9001],
               coolant_recool_us=1000.0,
               cool_ancilla_after_measure_us=1000.0,
               liz_pt_threshold_transversal=theta,
               liz_pt_target_transversal=None,
               liz_pt_max_iters=32, liz_pt_park="temp")
    E.compile_rounds_d(d, cfg, rounds=3, tp=tp)
    fired = [r for r in TRACE if r["fired"]]
    n = len(TRACE)
    halves = []
    for lo, hi in ((0, n // 2), (n // 2, n)):
        part = [r for r in fired if lo <= r["ms"] < hi]
        rate = len(part) / max(1, hi - lo)
        mean_c = (sum(r["cycles"] for r in part) / len(part)
                  if part else 0.0)
        halves.append((rate, mean_c))
    print(f"d={d} th={theta}: {n} MS, fired {len(fired)} "
          f"({len(fired)/n:.0%}) | 前半 fire {halves[0][0]:.0%} "
          f"cycles {halves[0][1]:.2f} | 後半 fire {halves[1][0]:.0%} "
          f"cycles {halves[1][1]:.2f}")
    return {"d": d, "theta": theta, "n_ms": n, "trace": list(TRACE)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [run(d, th) for d in (3, 5, 7) for th in (2.8, 1.5, 0.2)]
    out = OUT_DIR / "trace.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "t=1ms r=0 mainz N_max=32 temp, 3 rounds",
                   "runs": runs}, f)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
