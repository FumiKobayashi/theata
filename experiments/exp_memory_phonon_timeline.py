"""E-C9: メモリーゾーン加熱の時系列プロット (コンパイル replay のみ、Stim 不要)。

プロトコル (著者案 2026-08-23):
  全スタビライザ測定後に冷却 = cool_ancilla_measure_interval=24 (d=5 は
  24 stabilizer/round) でアンシラを LIZ 内冷却。メモリーゾーンの
  チェーン平均フォノン数を各スタビライザ測定完了時点でサンプリングし、
  「冷却 #k 直前」vs「冷却 #(k-1) 直後」のペア比較で正味加熱/round を出す。

実装方式:
  MovementPlanner をサブクラス化し、_annotate_gate (MEASURE 時) と
  _maybe_cool_after_measure / _maybe_auto_cool (冷却前後) でゾーン別
  フォノンスナップショットを記録する。コンパイラ本体は無変更。
  フォノン状態はラウンド間で引き継ぐ (state を毎ラウンド作り直すが
  phonon_axial/transversal を前ラウンド最終値で seed する)。
  ※既存の p_L 実験 (exp_cooling_pl 等) はラウンド毎にフォノンを 0 リセット
    してコンパイルしている。本実験は memory 蓄積の可視化が目的のため
    引き継ぎが物理的に正しい会計。

Usage:
    python experiments/exp_memory_phonon_timeline.py --rounds 10
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from ast_def import MeasureOp
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import (
    QCCDArchitecture, QCCDState, MovementPlanner, CompilerConfig,
)
from exp_distance_scan import DISTANCE_CIRCUITS
from exp_multi_round import placement_greedy_alt

MEMORY_ZONES = (1, 5)  # simple_mlm: T-M-T-L-T-M-T


class RecordingPlanner(MovementPlanner):
    """測定・冷却時点でゾーン別フォノンを記録する MovementPlanner。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeline = []
        self.measure_count = 0
        self.current_round = 0

    def _snapshot(self, state, tag):
        zones = {}
        for z, ions in state.zone_contents.items():
            if not ions:
                continue
            zones[z] = {
                'n_ions': len(ions),
                'mean_tr': sum(state.phonon_transversal.get(i, 0.0)
                               for i in ions) / len(ions),
                'mean_ax': sum(state.phonon_axial.get(i, 0.0)
                               for i in ions) / len(ions),
            }
        return {'tag': tag, 'round': self.current_round,
                'measure_index': self.measure_count, 'zones': zones}

    def _annotate_gate(self, op, state):
        result = super()._annotate_gate(op, state)
        if isinstance(op, MeasureOp) and op.targets:
            self.measure_count += 1
            self.timeline.append(self._snapshot(state, 'measure'))
        return result

    def _maybe_cool_after_measure(self, op, state):
        pre = self._snapshot(state, 'pre_cool')
        cool = super()._maybe_cool_after_measure(op, state)
        if cool is not None:
            self.timeline.append(pre)
            self.timeline.append(self._snapshot(state, 'post_cool'))
        return cool

    def _maybe_auto_cool(self, op, state):
        pre = self._snapshot(state, 'pre_cool')
        cool = super()._maybe_auto_cool(op, state)
        if cool is not None:
            self.timeline.append(pre)
            self.timeline.append(self._snapshot(state, 'post_cool'))
        return cool


def mem_mean(rec, key='mean_tr'):
    """メモリーゾーン (zone 1, 5) のイオン数加重平均フォノン。"""
    total, n = 0.0, 0
    for z in MEMORY_ZONES:
        zi = rec['zones'].get(z)
        if zi:
            total += zi[key] * zi['n_ions']
            n += zi['n_ions']
    return total / n if n else 0.0


def run_protocol(label, cfg_kwargs, rounds):
    forward, reverse, _ = DISTANCE_CIRCUITS[5]
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    cfg = CompilerConfig(**cfg_kwargs)
    planner = RecordingPlanner(arch, cfg)

    phonon_ax = phonon_tr = None
    zone_contents = None
    for r in range(1, rounds + 1):
        planner.current_round = r
        name = forward if r % 2 == 1 else reverse
        mod = __import__(name)
        dag = DAGBuilder(parse(mod.CIRCUIT), stabilizer_parallel=False).build()

        state = QCCDState(architecture=arch)
        state.phonon_params = dict(planner.phonon_params)
        if zone_contents is None:
            for ion, z in initial.items():
                state.place_ion_any(ion, z)
        else:
            for z in sorted(zone_contents):
                for ion in zone_contents[z]:
                    state.place_ion_any(ion, z)
        if phonon_ax:
            state.phonon_axial.update(phonon_ax)
            state.phonon_transversal.update(phonon_tr)

        planner.plan(dag, state=state)

        phonon_ax = dict(state.phonon_axial)
        phonon_tr = dict(state.phonon_transversal)
        zone_contents = {z: list(v) for z, v in state.zone_contents.items()}

    n_meas = planner.measure_count
    n_cools = sum(1 for rec in planner.timeline if rec['tag'] == 'post_cool')
    print(f"[{label}] measures={n_meas} cools={n_cools} "
          f"final mem mean_tr={mem_mean(planner.timeline[-1]):.3f}")
    return planner.timeline


def make_plots(timelines, rounds, out_dir, stamp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_COOL = '#2a78d6'    # anc24 (categorical slot 1)
    C_NOCOOL = '#eb6834'  # no cooling (slot 2)
    SURFACE = '#fcfcfb'
    INK = '#0b0b0b'
    INK2 = '#52514e'

    for key, unit, fname in (('mean_tr', 'transverse', 'transverse'),
                             ('mean_ax', 'axial', 'axial')):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 7.5),
            gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.32})
        fig.patch.set_facecolor(SURFACE)

        # --- Panel 1: memory mean phonon vs stabilizer measurement index ---
        ax1.set_facecolor(SURFACE)
        cool_label = next(k for k in timelines if k != 'no cooling')
        for label, color, ls in ((cool_label, C_COOL, '-'),
                                 ('no cooling', C_NOCOOL, '--')):
            tl = timelines[label]
            xs = [rec['measure_index'] for rec in tl if rec['tag'] == 'measure']
            ys = [mem_mean(rec, key) for rec in tl if rec['tag'] == 'measure']
            ax1.plot(xs, ys, ls, color=color, linewidth=2, label=label)
            ax1.annotate(label, xy=(xs[-1], ys[-1]), xytext=(6, 0),
                         textcoords='offset points', color=color,
                         fontsize=9, va='center')
        cool_tl = timelines[cool_label]
        cool_marks = [rec['measure_index'] for rec in cool_tl
                      if rec['tag'] == 'post_cool']
        if len(cool_marks) <= 40:
            for x in cool_marks:
                ax1.axvline(x, color=C_COOL, alpha=0.18, linewidth=1)
        ax1.set_xlabel('stabilizer measurement index', color=INK)
        ax1.set_ylabel(f'mean {unit} phonon $\\bar{{n}}$\n(memory zones)',
                       color=INK)
        ax1.set_title(
            f'Memory-zone heating, surface code d=5, {rounds} rounds '
            f'(cooling events marked)', color=INK, fontsize=11)
        ax1.grid(True, alpha=0.25, linewidth=0.6)
        ax1.margins(x=0.08)
        for s in ('top', 'right'):
            ax1.spines[s].set_visible(False)

        # --- Panel 2: pre-cool[k] vs post-cool[k-1] pairing (net heating/round) ---
        ax2.set_facecolor(SURFACE)
        pre = [rec for rec in cool_tl if rec['tag'] == 'pre_cool']
        post = [rec for rec in cool_tl if rec['tag'] == 'post_cool']
        ks, post_prev_v, pre_v = [], [], []
        for k in range(2, len(pre) + 1):
            ks.append(k)
            post_prev_v.append(mem_mean(post[k - 2], key))
            pre_v.append(mem_mean(pre[k - 1], key))
        few = len(ks) <= 30
        if few:
            for k, a, b in zip(ks, post_prev_v, pre_v):
                ax2.plot([k, k], [a, b], color=C_COOL, alpha=0.45,
                         linewidth=1.5)
        ax2.plot(ks, post_prev_v, 'o' if few else '-', color=C_COOL,
                 markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.8,
                 linewidth=1.5, label='after previous cooling')
        ax2.plot(ks, pre_v, 'o' if few else '-', color=C_COOL, markersize=7,
                 alpha=1.0 if few else 0.5, linewidth=1.5,
                 label='before next cooling')
        if few:
            for k, a, b in zip(ks, post_prev_v, pre_v):
                ax2.annotate(f'+{b - a:.2f}', xy=(k, b), xytext=(0, 6),
                             textcoords='offset points', ha='center',
                             fontsize=8, color=INK2)
        ax2.set_xlabel('cooling event k', color=INK)
        ax2.set_ylabel(f'memory mean {unit} $\\bar{{n}}$', color=INK)
        ax2.set_title('Net memory heating per round under ancilla-only '
                      'cooling (indirect cooling included)', color=INK,
                      fontsize=11)
        if few:
            ax2.set_xticks(ks)
        ax2.grid(True, alpha=0.25, linewidth=0.6)
        ax2.legend(frameon=False, fontsize=9, loc='upper left')
        for s in ('top', 'right'):
            ax2.spines[s].set_visible(False)

        png = out_dir / f'memory_phonon_d5_r{rounds}_{fname}_{stamp}.png'
        fig.savefig(png, dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"saved: {png}")


def main():
    ap = argparse.ArgumentParser(description='E-C9 memory phonon timeline')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--cool-us', type=float, default=400.0)
    ap.add_argument('--cool-interval', type=int, default=24,
                    help='何回の測定ごとにアンシラを冷却するか '
                         '(24 = ラウンド毎, 1 = 測定毎)')
    ap.add_argument('--out-dir', default='results/memory_phonon')
    args = ap.parse_args()

    cool_label = (f'cool ancilla / round' if args.cool_interval == 24
                  else f'cool ancilla / {args.cool_interval} meas')
    arms = {
        cool_label: {
            'cool_ancilla_after_measure_us': args.cool_us,
            'cool_ancilla_measure_interval': args.cool_interval,
        },
        'no cooling': {},
    }
    timelines = {label: run_protocol(label, kw, args.rounds)
                 for label, kw in arms.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    dump = {
        'meta': {'distance': 5, 'rounds': args.rounds,
                 'protocol': 'cool_ancilla_measure_interval=24',
                 'cool_us': args.cool_us, 'stamp': stamp,
                 'phonon_carryover_across_rounds': True},
        'timelines': timelines,
    }
    js = out_dir / f'memory_phonon_d5_r{args.rounds}_{stamp}.json'
    js.write_text(json.dumps(dump, indent=1), encoding='utf-8')
    print(f"saved: {js}")

    make_plots(timelines, args.rounds, out_dir, stamp)


if __name__ == '__main__':
    main()
