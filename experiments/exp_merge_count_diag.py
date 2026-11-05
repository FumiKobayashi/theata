"""Merge (= phonon averaging) counts per data qubit, by cooling strategy.

2026-11-05 meeting request (sensei): on fig4 (pl_vs_time_strategies),
N_max = 32 does not double the code-cycle time. Hypothesis: episodes end
early (threshold reached, or floor detection), so the number of
averaging events per data qubit stops growing. This script measures the
events directly: it patches QCCDState.merge_chains to count, for every
ion, how many merges it participates in during a 3-round compile, and
dumps per-qubit counts for the fig4 strategy arms (t = 1 ms,
n_res = 0, Mainz heating — same conditions as the figure).

Compile only, no Stim sampling.

Output: results/merge_counts_1105/merge_counts.json
Usage:  python experiments/exp_merge_count_diag.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import exp_cooling_distance_scan as E  # noqa: E402 (src/circuits paths)
import qccd_compiler as QC  # noqa: E402
from sim_params import load_time_parameters  # noqa: E402
from exp_distance_scan import DISTANCE_CIRCUITS  # noqa: E402
from parallel_estimator import ParallelEstimator  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "merge_counts_1105"

MERGES = Counter()          # ion_id -> participation count
_orig_merge = QC.QCCDState.merge_chains


def _counting_merge(self, zone_left, zone_right):
    for ion in (self.zone_contents.get(zone_left, [])
                + self.zone_contents.get(zone_right, [])):
        MERGES[ion] += 1
    return _orig_merge(self, zone_left, zone_right)


QC.QCCDState.merge_chains = _counting_merge

ANC1MS = {"cool_ancilla_after_measure_us": 1000.0}
RES = {"resident_coolant_ids": [9000, 9001], "coolant_recool_us": 1000.0,
       **ANC1MS}


def mpt(n):
    return {"predictive_cool_threshold_transversal": 1.5,
            "predictive_cool_max_iters": n, **ANC1MS}


def lpt(theta, iters=32):
    return {"liz_pt_threshold_transversal": theta,
            "liz_pt_target_transversal": None,
            "liz_pt_max_iters": iters, "liz_pt_park": "temp", **RES}


# fig4 の系列 + N_max 対照 (θ=1.5 の N4 vs N32: 早期終了の直接証拠)
ARMS = [
    ("no cooling", {}),
    ("LIZ cooling", ANC1MS),
    ("MPT N=1", mpt(1)),
    ("MPT N=4", mpt(4)),
    ("MPT N=16", mpt(16)),
    ("LPT th=2.8", lpt(2.8)),
    ("LPT th=1.5 N=4", lpt(1.5, iters=4)),
    ("LPT th=1.5", lpt(1.5)),
    ("LPT th=0.2", lpt(0.2)),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tp = load_time_parameters(str(ROOT / "params" / "mainz_heating.json"),
                              overrides={"cool": 1000.0})
    records = []
    for d in (3, 5, 7):
        fwd = __import__(DISTANCE_CIRCUITS[d][0])
        data_ids = sorted(fwd.DATA_QUBITS)
        anc_ids = sorted(fwd.ANCILLA_QUBITS)
        for label, cfg in ARMS:
            MERGES.clear()
            results = E.compile_rounds_d(d, dict(cfg), rounds=3, tp=tp)
            pe = ParallelEstimator(tp)
            wall_ms = sum(pe.estimate(r.timesteps).total_time / 1000
                          for r in results)
            rec = {
                "arm": label, "d": d, "rounds": 3,
                "wall_ms": wall_ms,
                "data_counts": {str(q): MERGES.get(q, 0) for q in data_ids},
                "ancilla_counts": {str(q): MERGES.get(q, 0)
                                   for q in anc_ids},
                "coolant_counts": {str(q): MERGES.get(q, 0)
                                   for q in (9000, 9001) if q in MERGES},
            }
            vals = [MERGES.get(q, 0) for q in data_ids]
            rec["data_mean"] = sum(vals) / len(vals)
            rec["data_max"] = max(vals)
            records.append(rec)
            print(f"d={d} {label:16s} data merges/qubit "
                  f"mean={rec['data_mean']:6.1f} max={rec['data_max']:4d} "
                  f"wall={wall_ms:7.1f}ms")
    out = OUT_DIR / "merge_counts.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "t=1ms r=0 mainz, 3 rounds, compile-only",
                   "records": records}, f, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
