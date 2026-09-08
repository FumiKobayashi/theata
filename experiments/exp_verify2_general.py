"""検証②: 一般符号 4 種で π→minpos→greedy_alt パイプラインを検証する。

検証① (d5 表面符号、9/3) の結論「π を順序生成器として使い、minpos 順 +
greedy_alt 再配置で snake +3〜6%」が、自然な人手順序 (snake) を持たない
一般符号で「H 行順 + greedy_alt」(現状基準) に勝つかを実測する。

符号 4 種: hgp_rep3 / hgp_hamming74 / bb_72_12_6 / many_hypercube_l2
  - 符号定義 (basis, support) は既存の circuits/<name>.py の DSL から抽出する
    (X 型 = MS anc data / Z 型 = MS data anc で判別)。ldpc / qldpc 不要。
  - 抽出の正しさは hgp_rep3 (numpy HGP 構成) と many_hypercube_l2 (JSON の
    Hx/Hz) の 2 つのグラウンドトゥルースに対して assert で検証し、さらに
    全符号で「抽出 → 再生成した DSL が既存ファイルと一致」を確認する。

回路生成の一般化: css_code_utils.build_css_stim_circuit は任意の stabilizer
順序リストを受けるので、d5_generator の d² 前提を経由せずに測定順序から
回路をインメモリ生成できる (make_module)。reverse はスタビライザ順 +
各 support の逆順 (d5 の交互走査規則と同じ)。minpos 順の CNOT 順は support
を π 位置昇順に並べる (走査の連続性)。

時間評価はロックステップ (従来) とイベント駆動 (schedule_events) を併記する。

ゾーン容量: memory = max(20, half + n_anc + slack) で符号サイズに追従
(--mem-slack, 既定 4 = 既存実験と同じ式)。使用した容量は出力に明記する。

Usage:
    python experiments/exp_verify2_general.py                 # 全 4 符号
    python experiments/exp_verify2_general.py --codes hgp_rep3
    python experiments/exp_verify2_general.py --sa-iters 30000 --rounds 3
"""

import sys
import json
import types
import argparse
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
SCRATCH_LIB = Path(
    r"C:\Users\ataro\AppData\Local\Temp\claude"
    r"\c--Users-ataro-Document-2025-autumn-B-thesis-qccd-simulator"
    r"\34e96d79-d0b4-4c52-840e-16c580e23b32\scratchpad\1D_Qubit_Mapping\src")
for p in (ROOT / 'src', ROOT / 'circuits', ROOT / 'experiments'):
    sys.path.insert(0, str(p))

import numpy as np


def _stub_ldpc():
    """css_code_utils の import に必要な ldpc を stub する
    (本実験は hgp_check_matrices / css_stabilizers / build_css_stim_circuit
    しか使わないため rank / distance 計算は不要)。"""
    try:
        import ldpc  # noqa: F401
        return
    except ImportError:
        pass
    m = types.ModuleType('ldpc')
    mod2 = types.ModuleType('ldpc.mod2')
    for n in ('rank', 'nullspace', 'pivot_rows'):
        setattr(mod2, n, lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('ldpc stub: not available')))
    cu = types.ModuleType('ldpc.code_util')
    cu.compute_exact_code_distance = lambda *a, **k: None
    cd = types.ModuleType('ldpc.codes')
    for n in ('hamming_code', 'rep_code', 'ring_code'):
        setattr(cd, n, lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('ldpc stub: not available')))
    sys.modules.update({'ldpc': m, 'ldpc.mod2': mod2,
                        'ldpc.code_util': cu, 'ldpc.codes': cd})
    try:
        import ortools  # noqa: F401
    except ImportError:
        om = types.ModuleType('ortools'); osat = types.ModuleType('ortools.sat')
        opy = types.ModuleType('ortools.sat.python')
        opy.cp_model = types.SimpleNamespace()
        sys.modules.update({'ortools': om, 'ortools.sat': osat,
                            'ortools.sat.python': opy})


_stub_ldpc()

from css_code_utils import (
    hgp_check_matrices, css_stabilizers, build_css_stim_circuit,
)
from serial_converter import SerialConverter  # path は css_code_utils が挿入済み
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from ast_def import GateOp, ResetOp, MeasureOp
from parallel_estimator import ParallelEstimator
from event_scheduler import schedule_events
from ion_timeline import build_ion_events, validate_order_against
from exp_multi_round import placement_greedy_alt
from exp_placement_bridge import order_to_initial
from exp_event_schedule import measure_hiding_stats

CODES = ['hgp_rep3', 'hgp_hamming74', 'bb_72_12_6', 'many_hypercube_l2']


# ==========================================
# 符号定義の抽出と検証
# ==========================================

def extract_stabilizers(mod):
    """既存回路モジュールから (basis, [support in CNOT order]) 列を抽出。

    X 型: MS の第 1 ターゲット = ancilla (CX anc→data)
    Z 型: MS の第 2 ターゲット = ancilla (CX data→anc)
    """
    anc = mod.ANCILLA_QUBITS[0]
    program = parse(mod.CIRCUIT)
    stabs, cur, basis = [], None, None
    for op in program.instructions:
        if isinstance(op, ResetOp) and list(op.targets) == [anc]:
            cur, basis = [], None
        elif isinstance(op, GateOp) and op.type == 'MS' and cur is not None:
            a, b = op.targets
            if anc not in (a, b):
                raise ValueError(f"MS without ancilla inside block: {op.targets}")
            if basis is None:
                basis = 'X' if a == anc else 'Z'
            cur.append(b if a == anc else a)
        elif isinstance(op, MeasureOp) and cur is not None:
            if basis is None:
                raise ValueError("stabilizer block without MS gates")
            stabs.append((basis, cur))
            cur, basis = None, None
    return stabs


def ground_truth_stabilizers(name):
    """抽出器の検証用グラウンドトゥルース (取れる符号のみ)。"""
    if name == 'hgp_rep3':
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)  # repetition(3)
        Hx, Hz = hgp_check_matrices(H, H)
        return css_stabilizers(Hx, Hz)
    if name == 'many_hypercube_l2':
        data = json.loads((ROOT / 'experiments' / 'code_params' /
                           'many_hypercube_l2_row_compact.json').read_text(encoding='utf-8'))
        Hx = np.array(data['matrices']['Hx'], dtype=np.uint8)
        Hz = np.array(data['matrices']['Hz'], dtype=np.uint8)
        return css_stabilizers(Hx, Hz)
    return None


def load_code(name):
    """回路モジュールから符号定義を抽出し、可能なら検証する。"""
    mod = __import__(name)
    stabs = extract_stabilizers(mod)
    n_data = len(mod.DATA_QUBITS)
    anc = mod.ANCILLA_QUBITS[0]

    gt = ground_truth_stabilizers(name)
    if gt is not None:
        assert [(b, sorted(s)) for b, s in stabs] == [(b, sorted(s)) for b, s in gt], \
            f"{name}: extracted stabilizers differ from ground truth"

    # 再生成 round-trip: 抽出列から作った DSL が既存モジュールと一致するか
    regen = SerialConverter().convert(build_css_stim_circuit(stabs, anc))
    orig_instr = [ln for ln in mod.CIRCUIT.splitlines()
                  if ln and not ln.startswith('#') and not ln.startswith('QUBIT')]
    regen_instr = [ln for ln in regen if ln and not ln.startswith('#')]
    assert orig_instr == regen_instr, \
        f"{name}: regenerated DSL differs from circuit file ({len(orig_instr)} vs {len(regen_instr)} lines)"

    return {'name': name, 'n_data': n_data, 'ancilla': anc, 'stabs': stabs}


# ==========================================
# 測定順序 → 回路モジュール (インメモリ)
# ==========================================

def make_module(name, stabs, n_data, anc, reverse=False):
    """スタビライザ順序リストから d5_reuse1 相当のインメモリモジュールを作る。"""
    ordered = ([(b, list(reversed(s))) for b, s in reversed(stabs)]
               if reverse else stabs)
    dsl = SerialConverter().convert(build_css_stim_circuit(ordered, anc))
    decls = [f"QUBIT {q} AT (0, {q})" for q in range(n_data)]
    decls.append(f"QUBIT {anc} AT (2, 0)")
    mod = types.ModuleType(name)
    mod.QUBIT_COUNT = n_data + 1
    mod.DATA_QUBITS = list(range(n_data))
    mod.ANCILLA_QUBITS = [anc]
    mod.CIRCUIT = "\n".join(decls) + "\n\n" + "\n".join(dsl) + "\n"
    sys.modules[name] = mod
    return name


def minpos_order(stabs, pi_order):
    """π 上の最小位置で生成元をソートし (tie: 最大位置)、各 support も
    π 位置昇順に並べる (走査の連続性)。検証① の minpos 規則の一般符号版。"""
    pos = {int(q): i for i, q in enumerate(pi_order)}
    ordered = [(b, sorted(s, key=lambda q: pos[q])) for b, s in stabs]
    ordered.sort(key=lambda bs: (min(pos[q] for q in bs[1]),
                                 max(pos[q] for q in bs[1])))
    return ordered


# ==========================================
# 評価 (ロックステップ + イベント駆動)
# ==========================================

def evaluate(fwd_name, rev_name, initial, rounds, mem_cap):
    fwd = sys.modules.get(fwd_name) or __import__(fwd_name)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    pe = ParallelEstimator()
    placement, zone_contents = initial, None
    tot = {'shuttles': 0, 'swaps': 0, 'splits': 0, 'merges': 0}
    lock_ms, event_ms, gate_tr = [], [], []
    hidden_us_r1 = None
    for r in range(1, rounds + 1):
        mod_name = fwd_name if r % 2 == 1 else rev_name
        mod = sys.modules.get(mod_name) or __import__(mod_name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents)
                         for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=CompilerConfig()).compile(dag, placement)
        for k in tot:
            tot[k] += cr.stats.get(k, 0)
        lock_ms.append(pe.estimate(cr.timesteps).total_time / 1000)
        intervals, t_end = schedule_events(cr.flat_ops)
        event_ms.append(t_end / 1000)
        if r == 1:
            # 整合性検証 + 測定隠蔽量 (round 1 のみ、コスト削減)
            ev = build_ion_events(intervals)
            validate_order_against(ev, cr.flat_ops)
            _, _, hidden_us_r1 = measure_hiding_stats(intervals)
        for g in (cr.phonon_data or {}).get('at_gates', []):
            gate_tr.append(max(p[1] for p in g['phonons'].values()))
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
    arr = np.array(gate_tr)
    return {
        'transport': tot,
        'lock_ms': lock_ms, 'event_ms': event_ms,
        'nbar_cap': float(arr.max()), 'nbar_avg': float(arr.mean()),
        'hidden_us_r1': hidden_us_r1,
    }


CSV_ROWS = []


def report(code_name, label, schedule, ev, span=None):
    tot = ev['transport']
    print(f"  {label:44s} | {tot['shuttles']:5d} {tot['swaps']:5d} {tot['splits']:5d} "
          f"{tot['merges']:5d} | {sum(ev['lock_ms']):8.1f} {sum(ev['event_ms']):8.1f} | "
          f"{ev['nbar_cap']:6.2f} {ev['nbar_avg']:5.2f}", flush=True)
    CSV_ROWS.append({
        'code': code_name, 'schedule': schedule, 'placement': label.strip(),
        'C_span': '' if span is None else int(span),
        'n_shuttle': tot['shuttles'], 'n_swap': tot['swaps'],
        'n_split': tot['splits'], 'n_merge': tot['merges'],
        'T_lock_ms': round(sum(ev['lock_ms']), 1),
        'T_event_ms': round(sum(ev['event_ms']), 1),
        'round_lock_ms': ';'.join(f"{t:.1f}" for t in ev['lock_ms']),
        'round_event_ms': ';'.join(f"{t:.1f}" for t in ev['event_ms']),
        'hidden_us_r1': round(ev['hidden_us_r1'], 0),
        'nbar_cap': round(ev['nbar_cap'], 2), 'nbar_avg': round(ev['nbar_avg'], 2),
    })


def run_code(code, strategies_fn, rounds, mem_slack):
    name, n_data, anc, stabs = code['name'], code['n_data'], code['ancilla'], code['stabs']
    half = (n_data + 1) // 2
    mem_cap = max(20, half + 1 + mem_slack)
    print(f"\n=== {name}  (n_data={n_data}, stabs={len(stabs)}, "
          f"weights {min(len(s) for _, s in stabs)}-{max(len(s) for _, s in stabs)}, "
          f"memory_capacity={mem_cap}) ===")
    print(f"  {'row':44s} | {'shut':>5s} {'swap':>5s} {'split':>5s} {'merge':>5s} | "
          f"{'T_lock':>8s} {'T_evt':>8s} | {'cap':>6s} {'avg':>5s}")

    # --- baseline: H 行順 + greedy_alt ---
    fwd = make_module(f'_v2_{name}_hrow_fwd', stabs, n_data, anc, reverse=False)
    rev = make_module(f'_v2_{name}_hrow_rev', stabs, n_data, anc, reverse=True)
    init = placement_greedy_alt(sys.modules[fwd].DATA_QUBITS,
                                sys.modules[fwd].ANCILLA_QUBITS, sys.modules[fwd])
    report(name, 'hrow + greedy_alt (baseline)', 'hrow',
           evaluate(fwd, rev, init, rounds, mem_cap))

    # --- π 戦略ごと: minpos 順序 + (π 配置 / greedy_alt 再配置) ---
    from qccd_mapping.hypergraph import StabilizerHypergraph
    from qccd_mapping.codes.base import CSSCode
    from qccd_mapping.placement import order_from_placement

    Hx = np.zeros((sum(1 for b, _ in stabs if b == 'X'), n_data), dtype=np.uint8)
    Hz = np.zeros((sum(1 for b, _ in stabs if b == 'Z'), n_data), dtype=np.uint8)
    ix = iz = 0
    for b, s in stabs:
        if b == 'X':
            Hx[ix, s] = 1; ix += 1
        else:
            Hz[iz, s] = 1; iz += 1
    hg = StabilizerHypergraph.from_css_code(
        CSSCode(name=name, H_X=Hx, H_Z=Hz, coords=None))

    import time
    for strat_name, strat in strategies_fn():
        t0 = time.time()
        p = strat.solve(hg)
        print(f"  [{strat_name}] solve {time.time()-t0:.1f}s", flush=True)
        pi = order_from_placement(p)
        ordered = minpos_order(stabs, pi)
        fwd = make_module(f'_v2_{name}_{strat_name}_fwd', ordered, n_data, anc)
        rev = make_module(f'_v2_{name}_{strat_name}_rev', ordered, n_data, anc, reverse=True)
        span = hg.total_span(p)
        # (a) π 配置そのまま
        init_pi = order_to_initial(pi, [anc])
        report(name, f'{strat_name} (π + minpos)', 'minpos',
               evaluate(fwd, rev, init_pi, rounds, mem_cap), span)
        # (b) 本命: minpos 回路に greedy_alt 再配置
        m = sys.modules[fwd]
        init_ga = placement_greedy_alt(m.DATA_QUBITS, m.ANCILLA_QUBITS, m)
        report(name, f'  greedy_alt on {strat_name} minpos', 'minpos',
               evaluate(fwd, rev, init_ga, rounds, mem_cap), span)


def main():
    ap = argparse.ArgumentParser(description='検証②: 一般符号での π→minpos→greedy_alt')
    ap.add_argument('--codes', nargs='*', default=CODES)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sa-iters', type=int, default=30000)
    ap.add_argument('--mem-slack', type=int, default=4)
    ap.add_argument('--lib', default=str(SCRATCH_LIB),
                    help='1D_Qubit_Mapping/src へのパス')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.lib).resolve()))
    from qccd_mapping.placement import (
        SequentialPlacement, SpectralPlacement, LocalSearchMinLA,
        StabilizerSpanPlacement, QCCDAwarePlacement,
        placement_from_order,
    )

    # ⚠ ライブラリ不具合の回避 (要先生報告):
    # nx.spectral_ordering (tracemin_pcg) は BB 符号のような固有値縮退の強い
    # 相互作用グラフで収束せず、事実上ハングする (bb_72_12_6 で 5 時間超)。
    # n<=144 では密行列の固有分解で十分なので、Fiedler ベクトルを
    # np.linalg.eigh で決定的に計算する実装に差し替える。
    # LocalSearchMinLA も内部で SpectralPlacement をシードに使うため、
    # クラスメソッドの monkeypatch で両方に効かせる。
    import networkx as nx

    def _dense_spectral_solve(self, hypergraph, rng=None):
        G = hypergraph.interaction_graph()
        n = hypergraph.n
        if G.number_of_edges() == 0:
            return placement_from_order(list(range(n)))
        L = nx.laplacian_matrix(G, nodelist=list(range(n)),
                                weight='weight').toarray().astype(float)
        _, vecs = np.linalg.eigh(L)
        return placement_from_order(list(np.argsort(vecs[:, 1])))

    SpectralPlacement.solve = _dense_spectral_solve

    def strategies_fn():
        return [
            ('sequential', SequentialPlacement()),
            ('spectral', SpectralPlacement()),
            ('minla_localsearch', LocalSearchMinLA(iterations=5000, seed=0)),
            ('stabilizer_span_SA', StabilizerSpanPlacement(iterations=args.sa_iters, seed=0)),
            ('qccd_aware_SA', QCCDAwarePlacement(iterations=args.sa_iters, seed=0)),
        ]

    for name in args.codes:
        code = load_code(name)
        print(f"[OK] {name}: extraction validated "
              f"({len(code['stabs'])} stabilizers, round-trip DSL match)")
        run_code(code, strategies_fn, args.rounds, args.mem_slack)

    out_dir = ROOT / 'results' / 'verify2_general'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    import csv
    out = out_dir / f'verify2_general_{ts}.csv'
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_ROWS[0].keys()))
        w.writeheader()
        w.writerows(CSV_ROWS)
    print(f'\ncsv: {out}')


if __name__ == '__main__':
    main()
