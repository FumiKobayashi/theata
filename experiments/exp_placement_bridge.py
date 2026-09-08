"""1D_Qubit_Mapping (先生の静的配置ライブラリ) との橋渡しプローブ。

先生の各 placement 戦略で得た 1D 順序 π を本コンパイラの initial_placement
(memory zone 1 / 5 に 2 分) に変換し、コンパイラを cost oracle として
transport 回数・T_round・ゲート時フォノン (cap/avg) を評価する。
H_X/H_Z は circuits/d5_generator.get_stabilizers から直接構築する
(Stim との番号付け対応を不要にするため)。

--schedule で回路 (スタビライザ測定順序) の扱いを選ぶ:
  snake : 既存の蛇行順回路 (d5_reuse1 等) を固定して配置だけ差し替える
  pi    : 各 π に整合する測定順序 (support の π 上の最小位置でソート) で
          回路を生成し直す (generate_circuit_from_order、CNOT 順は表面符号の
          hook-error 回避規則)。配置 + 順序の対等な比較
          ⚠ この規則は「チェーン左端 = zone 1 の最深部」から走査するため
          zone レイアウトに不利 (9/1 洗い出し)。歴史的比較用に残す
  liz   : LIZ 基準の貪欲順序 (9/1 の逆向き貪欲法)。各生成元を「最深メンバー
          の LIZ 端からの深さ」で浅い順に、左右 zone 交互に測る。
          各 π について (a) π 配置 + liz 順序、(b) liz 順序回路に greedy_alt
          再配置 (= π を順序生成器として使う本命パイプライン) を評価
  both  : snake + pi
  all   : snake + pi + liz

ortools / ldpc が未インストールでもヒューリスティック配置は動くよう、
import 失敗時は stub を差し込む (ExactSpanPlacement / optimal_route は不可)。

Usage:
    python experiments/exp_placement_bridge.py --lib ../1D_Qubit_Mapping/src --schedule both
"""

import sys
import types
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / 'src', ROOT / 'circuits', ROOT / 'experiments'):
    sys.path.insert(0, str(p))

import numpy as np


def _stub_optional_deps():
    """ortools / ldpc が無い環境向けの stub。"""
    try:
        import ortools  # noqa: F401
    except ImportError:
        m = types.ModuleType('ortools'); s = types.ModuleType('ortools.sat')
        py = types.ModuleType('ortools.sat.python'); py.cp_model = types.SimpleNamespace()
        sys.modules.update({'ortools': m, 'ortools.sat': s, 'ortools.sat.python': py})
    try:
        import ldpc  # noqa: F401
    except ImportError:
        m = types.ModuleType('ldpc'); c = types.ModuleType('ldpc.codes')
        for n in ('hamming_code', 'rep_code', 'ring_code'):
            setattr(c, n, lambda *a, **k: None)
        sys.modules.update({'ldpc': m, 'ldpc.codes': c})


def css_from_generator(d):
    """本リポジトリの surface-code 生成器から CSSCode を構築 (data id = r*d + c)。"""
    import d5_generator
    from qccd_mapping.codes.base import CSSCode
    stabs = d5_generator.get_stabilizers(d)
    xs = [q for t, q, _ in stabs if t == 'X']
    zs = [q for t, q, _ in stabs if t == 'Z']
    H_X = np.zeros((len(xs), d * d), dtype=np.uint8)
    H_Z = np.zeros((len(zs), d * d), dtype=np.uint8)
    for r, q in enumerate(xs):
        H_X[r, q] = 1
    for r, q in enumerate(zs):
        H_Z[r, q] = 1
    coords = {i: (float(i % d), float(i // d)) for i in range(d * d)}  # (x, y)
    return CSSCode(name=f'rsc_d{d}_ours', H_X=H_X, H_Z=H_Z, coords=coords)


def order_to_initial(order, ancilla_ids, left=1, right=5, ancilla_zone=5):
    """1D 順序 (data のみ) → {ion: zone}。dict 挿入順 = チェーン内順序。
    左半分 → zone left (LIZ 側 = 末尾)、右半分 → zone right (LIZ 側 = 先頭)。
    アンシラは ancilla_zone の LIZ 側の端。"""
    order = [int(q) for q in order]
    half = len(order) // 2
    initial = {}
    for q in order[:half]:
        initial[q] = left
    if ancilla_zone == left:
        for a in ancilla_ids:
            initial[a] = left
    else:
        for a in ancilla_ids:
            initial[a] = right
    for q in order[half:]:
        initial[q] = right
    return initial


def pi_schedule_modules(d, order, tag):
    """π に整合する測定順序で forward / reverse 回路をインメモリ生成し、
    (fwd_name, rev_name) を返す。順序 = support の π 上の最小位置 (tie: 最大位置)。"""
    import d5_generator as gen
    pos = {int(q): i for i, q in enumerate(order)}
    stabs = [(t, sorted(dq)) for t, dq, _ in gen.get_stabilizers(d)]
    stabs.sort(key=lambda s: (min(pos[q] for q in s[1]), max(pos[q] for q in s[1])))
    fwd_str, ordered, anc = gen.generate_circuit_from_order(d, stabs, 1, reverse=False)
    rev_str, _, _ = gen.generate_circuit_from_order(d, stabs, 1, reverse=True)
    fwd_name, rev_name = f'_bridge_{tag}_fwd', f'_bridge_{tag}_rev'
    gen.make_circuit_module(fwd_name, d, fwd_str, anc, ordered)
    gen.make_circuit_module(rev_name, d, rev_str, anc)
    return fwd_name, rev_name


def liz_greedy_order(d, order, alternate=True):
    """LIZ 基準の貪欲順序規則 (9/1 の逆向き貪欲法)。

    π (data の 1D 順序) を order_to_initial と同じ規則で 2 分したとき、
    zone 1 の LIZ 側端 = π 位置 half-1、zone 5 の LIZ 側端 = π 位置 half。
    depth(q) = q の LIZ 側端からの距離。各生成元 S のコストは最深メンバー
    (LIFO 的な取り出しで退避回数を決めるのは最深メンバー — §3.2 spanL_max
    が正の相関 +0.83 だった知見) とし、S を最深メンバーの側に帰属させて
    「浅い生成元から」測る順序を返す。

    alternate=True : 左右 zone 交互 (9/1 提案そのまま。9/3 実測で
        走査の連続性を壊し大幅悪化 → 棄却)
    alternate=False: 片側ずつ LIZ から外向きに走査 (lizseq)。連続性を
        保ったまま浅い生成元から始める切り分け用変種
    """
    import d5_generator as gen
    pos = {int(q): i for i, q in enumerate(order)}
    half = len(order) // 2

    def depth(q):
        p = pos[q]
        return (half - 1 - p) if p < half else (p - half)

    stabs = [(t, sorted(dq)) for t, dq, _ in gen.get_stabilizers(d)]
    sides = ([], [])  # 0: zone 1 (左半分), 1: zone 5 (右半分)
    for s in stabs:
        dq = s[1]
        deepest = max(dq, key=depth)
        key = (max(depth(q) for q in dq), sum(depth(q) for q in dq))
        sides[0 if pos[deepest] < half else 1].append((key, s))
    for b in sides:
        b.sort(key=lambda x: x[0])
    if not alternate:
        return [s for b in sides for _, s in b]
    ordered, idx = [], [0, 0]
    turn = 0 if (sides[0] and (not sides[1] or sides[0][0][0] <= sides[1][0][0])) else 1
    while idx[0] < len(sides[0]) or idx[1] < len(sides[1]):
        if idx[turn] >= len(sides[turn]):
            turn = 1 - turn
        ordered.append(sides[turn][idx[turn]][1])
        idx[turn] += 1
        turn = 1 - turn
    return ordered


def order_schedule_modules(d, ordered, tag):
    """生成元の測定順序 ordered から forward / reverse 回路をインメモリ生成。"""
    import d5_generator as gen
    fwd_str, resolved, anc = gen.generate_circuit_from_order(d, ordered, 1, reverse=False)
    rev_str, _, _ = gen.generate_circuit_from_order(d, ordered, 1, reverse=True)
    fwd_name, rev_name = f'_bridge_{tag}_fwd', f'_bridge_{tag}_rev'
    gen.make_circuit_module(fwd_name, d, fwd_str, anc, resolved)
    gen.make_circuit_module(rev_name, d, rev_str, anc)
    return fwd_name, rev_name


def evaluate(initial, d, rounds, fwd_name=None, rev_name=None):
    """本コンパイラを cost oracle として transport / 時間 / フォノンを返す。"""
    from qccd_parser import parse
    from dag_builder import DAGBuilder
    from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
    from parallel_estimator import ParallelEstimator
    from exp_distance_scan import DISTANCE_CIRCUITS

    if fwd_name is None:
        fwd_name, rev_name, _ = DISTANCE_CIRCUITS[d]
    fwd = __import__(fwd_name)
    half = (len(fwd.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    pe = ParallelEstimator()
    placement, zone_contents = initial, None
    tot = {'shuttles': 0, 'swaps': 0, 'splits': 0, 'merges': 0}
    t_ms, gate_tr = [], []
    for r in range(1, rounds + 1):
        mod = __import__(fwd_name if r % 2 == 1 else rev_name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()
        if zone_contents is not None:
            placement = {ion: z for z in sorted(zone_contents) for ion in zone_contents[z]}
        cr = QCCDCompiler(arch, config=CompilerConfig()).compile(dag, placement)
        for k in tot:
            tot[k] += cr.stats.get(k, 0)
        t_ms.append(pe.estimate(cr.timesteps).total_time / 1000)
        for g in (cr.phonon_data or {}).get('at_gates', []):
            gate_tr.append(max(p[1] for p in g['phonons'].values()))
        placement, zone_contents = cr.final_placement, cr.final_zone_contents
    arr = np.array(gate_tr)
    return tot, t_ms, float(arr.max()), float(arr.mean())


CSV_ROWS = []  # --csv 用に蓄積


def _print_row(name, span, inter, tot, t_ms, cap, avg, schedule=''):
    print(f"{name:34s} {span:7.0f} {inter:6.0f} | {tot['shuttles']:5d} {tot['swaps']:5d} "
          f"{tot['splits']:5d} {tot['merges']:5d} | {sum(t_ms):8.1f} | {cap:5.2f} {avg:5.2f}")
    CSV_ROWS.append({
        'schedule': schedule, 'placement': name.strip(),
        'C_span': '' if span != span else int(span),
        'C_intervening': '' if inter != inter else int(inter),
        'n_shuttle': tot['shuttles'], 'n_swap': tot['swaps'],
        'n_split': tot['splits'], 'n_merge': tot['merges'],
        'T_total_ms': round(sum(t_ms), 1),
        'round_ms': ';'.join(f'{t:.1f}' for t in t_ms),
        'nbar_cap': round(cap, 2), 'nbar_avg': round(avg, 2),
    })


def _write_csv(path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:  # BOM: Excel が UTF-8 と認識
        w = csv.DictWriter(f, fieldnames=list(CSV_ROWS[0].keys()))
        w.writeheader()
        w.writerows(CSV_ROWS)
    print(f'\n  csv: {path}')


def main():
    ap = argparse.ArgumentParser(description='placement bridge probe (1D_Qubit_Mapping → compiler)')
    ap.add_argument('--lib', required=True, help='1D_Qubit_Mapping/src へのパス')
    ap.add_argument('--distance', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sa-iters', type=int, default=30000)
    ap.add_argument('--schedule',
                    choices=['snake', 'pi', 'liz', 'lizseq', 'hrow', 'both', 'all'],
                    default='both')
    ap.add_argument('--family', choices=['default', 'boustro'], default='default',
                    help='比較する配置戦略の族。boustro = 牛耕式の変種比較 '
                         '(plain / 片端編み込み / 両端編み込み × 鏡映 2 向き)。'
                         '符号構築の鏡映で市松位相と開始方向の噛み合いが変わり '
                         'span の優劣が入れ替わるため両向きを明示 (9/3 知見)')
    ap.add_argument('--csv', default=None,
                    help='結果を CSV に書き出す (例: results/placement_bridge/run.csv)。'
                         'Excel でそのまま開ける (UTF-8 BOM 付き)')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.lib).resolve()))
    _stub_optional_deps()
    from qccd_mapping.hypergraph import StabilizerHypergraph
    from qccd_mapping.placement import (
        SequentialPlacement, SpectralPlacement, LocalSearchMinLA,
        StabilizerSpanPlacement, QCCDAwarePlacement,
        BoustrophedonPlacement, BoustrophedonAdaptivePlacement,
        order_from_placement, placement_from_order,
    )
    from exp_multi_round import placement_greedy_alt
    from exp_distance_scan import DISTANCE_CIRCUITS

    d = args.distance
    code = css_from_generator(d)
    hg = StabilizerHypergraph.from_css_code(code)
    fwd = __import__(DISTANCE_CIRCUITS[d][0])
    ancilla_ids = list(fwd.ANCILLA_QUBITS)
    print(f'{code}  hyperedges={len(hg.edges)}  rounds={args.rounds}  schedule={args.schedule}')

    init_ga = placement_greedy_alt(fwd.DATA_QUBITS, fwd.ANCILLA_QUBITS, fwd)
    order_ga = [q for q, z in init_ga.items() if q not in ancilla_ids and z == 1] + \
               [q for q, z in init_ga.items() if q not in ancilla_ids and z == 5]
    p_ga = placement_from_order(order_ga)

    if args.family == 'boustro':
        from qccd_mapping.placement.geometry_aware import (
            boustrophedon_order, boustrophedon_last_pair_interleaved_order,
            boustrophedon_both_ends_interleaved_order, _BOTH_ENDS_CONFIGS,
        )
        from qccd_mapping.placement import placement_from_order as pfo

        def best_both_ends(coords):
            orders = [boustrophedon_both_ends_interleaved_order(coords, a, b)
                      for a, b in _BOTH_ENDS_CONFIGS]
            return min(orders, key=lambda o: hg.total_span(pfo(o)))

        d_max = max(y for _, y in code.coords.values())
        flipped = {q: (x, d_max - y) for q, (x, y) in code.coords.items()}
        strategies = []
        for tag, coords in [('', code.coords), ('_flip', flipped)]:
            strategies += [
                (f'boustro_plain{tag}', boustrophedon_order(coords)),
                (f'boustro_single_end{tag}',
                 boustrophedon_last_pair_interleaved_order(coords)),
                (f'boustro_both_ends{tag}', best_both_ends(coords)),
            ]
        pis = [(name, pfo(order)) for name, order in strategies]
    else:
        strategies = [
            ('sequential', SequentialPlacement()),
            ('spectral', SpectralPlacement()),
            ('minla_localsearch', LocalSearchMinLA(iterations=5000, seed=0)),
            ('stabilizer_span_SA', StabilizerSpanPlacement(iterations=args.sa_iters, seed=0)),
            ('qccd_aware_SA', QCCDAwarePlacement(iterations=args.sa_iters, seed=0)),
            ('boustrophedon', BoustrophedonPlacement(code.coords)),
            ('boustrophedon_adaptive', BoustrophedonAdaptivePlacement(code.coords)),
        ]
        pis = [(name, strat.solve(hg)) for name, strat in strategies]

    header = (f"\n{'strategy':34s} {'C_span':>7s} {'C_int':>6s} | {'shut':>5s} {'swap':>5s} "
              f"{'split':>5s} {'merge':>5s} | {'T ms':>8s} | {'cap':>5s} {'avg':>5s}")

    if args.schedule in ('snake', 'both', 'all'):
        print("\n[schedule = snake (既存回路固定、配置のみ差し替え)]" + header)
        _print_row('greedy_alt (ours)', hg.total_span(p_ga), hg.total_intervening_cost(p_ga),
                   *evaluate(init_ga, d, args.rounds), schedule='snake')
        for name, p in pis:
            init = order_to_initial(order_from_placement(p), ancilla_ids)
            _print_row(name, hg.total_span(p), hg.total_intervening_cost(p),
                       *evaluate(init, d, args.rounds), schedule='snake')

    if args.schedule in ('pi', 'both', 'all'):
        print("\n[schedule = pi (π 整合の測定順序で回路を生成し直す)]" + header)
        _print_row('greedy_alt + snake (reference)', hg.total_span(p_ga),
                   hg.total_intervening_cost(p_ga), *evaluate(init_ga, d, args.rounds),
                   schedule='snake')
        for name, p in pis:
            order = order_from_placement(p)
            fwd_name, rev_name = pi_schedule_modules(d, order, name)
            init = order_to_initial(order, ancilla_ids)
            _print_row(f'{name} (π + π-schedule)', hg.total_span(p),
                       hg.total_intervening_cost(p),
                       *evaluate(init, d, args.rounds, fwd_name, rev_name),
                       schedule='pi')
            # 同じ π 回路に greedy_alt 配置を適用: 順序の効果と配置の効果を分離
            mod = __import__(fwd_name)
            init_ga_pi = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
            _print_row(f'  greedy_alt on {name} π-schedule', float('nan'), float('nan'),
                       *evaluate(init_ga_pi, d, args.rounds, fwd_name, rev_name),
                       schedule='pi')

    if args.schedule in ('liz', 'all'):
        print("\n[schedule = liz (LIZ 基準の貪欲順序: 浅い生成元から左右交互)]" + header)
        _print_row('greedy_alt + snake (reference)', hg.total_span(p_ga),
                   hg.total_intervening_cost(p_ga), *evaluate(init_ga, d, args.rounds),
                   schedule='snake')
        for name, p in pis:
            order = order_from_placement(p)
            ordered = liz_greedy_order(d, order)
            fwd_name, rev_name = order_schedule_modules(d, ordered, f'liz_{name}')
            init = order_to_initial(order, ancilla_ids)
            _print_row(f'{name} (π + liz-schedule)', hg.total_span(p),
                       hg.total_intervening_cost(p),
                       *evaluate(init, d, args.rounds, fwd_name, rev_name),
                       schedule='liz')
            # 本命パイプライン: π → liz 順序 → greedy_alt 再配置
            mod = __import__(fwd_name)
            init_ga_liz = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
            _print_row(f'  greedy_alt on {name} liz-schedule', float('nan'), float('nan'),
                       *evaluate(init_ga_liz, d, args.rounds, fwd_name, rev_name),
                       schedule='liz')

    if args.schedule in ('lizseq', 'all'):
        print("\n[schedule = lizseq (LIZ から外向き片側ずつ: 連続性を保つ変種)]" + header)
        _print_row('greedy_alt + snake (reference)', hg.total_span(p_ga),
                   hg.total_intervening_cost(p_ga), *evaluate(init_ga, d, args.rounds),
                   schedule='snake')
        for name, p in pis:
            order = order_from_placement(p)
            ordered = liz_greedy_order(d, order, alternate=False)
            fwd_name, rev_name = order_schedule_modules(d, ordered, f'lizseq_{name}')
            init = order_to_initial(order, ancilla_ids)
            _print_row(f'{name} (π + lizseq)', hg.total_span(p),
                       hg.total_intervening_cost(p),
                       *evaluate(init, d, args.rounds, fwd_name, rev_name),
                       schedule='lizseq')
            mod = __import__(fwd_name)
            init_ga_ls = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
            _print_row(f'  greedy_alt on {name} lizseq', float('nan'), float('nan'),
                       *evaluate(init_ga_ls, d, args.rounds, fwd_name, rev_name),
                       schedule='lizseq')

    if args.schedule in ('hrow', 'all'):
        # 一般符号の現状基準に相当: H の行順 (get_stabilizers の生成順) の
        # 回路 + greedy_alt 配置
        import d5_generator as gen
        print("\n[schedule = hrow (H 行順 + greedy_alt: 一般符号の現状基準の相当物)]" + header)
        stabs = [(t, sorted(dq)) for t, dq, _ in gen.get_stabilizers(d)]
        fwd_name, rev_name = order_schedule_modules(d, stabs, 'hrow')
        mod = __import__(fwd_name)
        init_hrow = placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)
        _print_row('greedy_alt on H-row order', float('nan'), float('nan'),
                   *evaluate(init_hrow, d, args.rounds, fwd_name, rev_name),
                   schedule='hrow')

    if args.csv:
        _write_csv(args.csv)


if __name__ == '__main__':
    main()
