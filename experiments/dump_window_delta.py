"""測定窓ごとの per-ion 増分 Δtr の分布ダンプ (2026-10-24)。

予測トリガの充足可能性チェック: 予測 = 現在値 + 次窓 Δ なので、
Δ > θ の窓では何回冷やしても予測を θ 未満にできず、トリガが
N_max 回フル発動する (飽和域)。θ グリッドの各値がどちらの領域かを
d = 3/5/7 で判定する。

Δ は pass 1 (トリガのみ無効の同一設定) のコンパイルが記録する決定論値
(モデル上の加熱は確率でなく決定論: per-op 加熱 + 時間比例 background)。
対象 = 各窓の data_ions (次窓で MS に参加するデータ量子ビット) の delta_tr。

Usage:
    python experiments/dump_window_delta.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np

from dag_builder import DAGBuilder
from qccd_parser import parse
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from sim_params import load_time_parameters
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt

ROUNDS = 3
THETAS = [0.7, 1.4, 1.5, 2.1, 2.8, 3.5, 4.2, 5.6]

# 正準スイープ設定 (run_sweeps_1024.ps1 と同一、t = 1 ms, n̄_res = 0)
CFG = dict(
    resident_coolant_ids=[9000, 9001],
    coolant_recool_us=1000.0,
    predictive_cool_threshold_transversal=1.5,   # pass 1 は本トリガ無効なので Δ は θ 非依存
    predictive_cool_max_iters=16,
    cool_ancilla_after_measure_us=1000.0,
)
OVERRIDES = {
    'cool': 1000.0,
    'phonon_cool_residual_axial': 0.0,
    'phonon_cool_residual_transversal': 0.0,
    'phonon_cool_residual_transversal2': 0.0,
}


def window_deltas_for_d(d, tp):
    forward, reverse, _ = DISTANCE_CIRCUITS[d]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)
    cfg = CompilerConfig(**CFG)
    liz_id = arch.get_liz_zones()[0].id
    initial = dict(initial)
    for cid in cfg.resident_coolant_ids:
        initial[cid] = liz_id

    deltas = []
    placement, zone_contents = initial, None
    for r in range(1, ROUNDS + 1):
        name = forward if r % 2 == 1 else reverse
        mod = __import__(name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        compiler = QCCDCompiler(arch, config=cfg, time_params=tp)
        cr = compiler.compile(dag, placement)
        table = compiler._planner.window_table or []
        for win in table:
            for ion in win['data_ions']:
                deltas.append(win['delta_tr'].get(ion, 0.0))
        placement = cr.final_placement
        zone_contents = cr.final_zone_contents
    return np.array(deltas)


def main():
    tp = load_time_parameters('params/mainz_heating.json', overrides=OVERRIDES)
    print("per-window per-data-ion Δtr (pass 1, canonical sweep config, "
          f"{ROUNDS} rounds)\n")
    header = (f"{'d':>2} {'n':>5} {'median':>7} {'mean':>7} {'p95':>7} "
              f"{'max':>7}  " + " ".join(f"P(Δ>{t:g})" for t in THETAS))
    print(header)
    for d in [3, 5, 7]:
        arr = window_deltas_for_d(d, tp)
        fracs = [(arr > th).mean() for th in THETAS]
        print(f"{d:>2} {arr.size:>5} {np.median(arr):7.3f} {arr.mean():7.3f} "
              f"{np.percentile(arr, 95):7.3f} {arr.max():7.3f}  "
              + " ".join(f"{f:8.3f}" for f in fracs))


if __name__ == "__main__":
    main()
