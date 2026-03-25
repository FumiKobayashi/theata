"""
Zone Visualizer: TimeStep ごとのイオン配置をテキストで可視化する

使い方:
    from zone_visualizer import ZoneVisualizer

    vis = ZoneVisualizer(architecture)
    # 全TimeStepをテキスト出力
    vis.print_all(timesteps, initial_placement)
    # 特定のTimeStep範囲を出力
    vis.print_range(timesteps, initial_placement, start=0, end=10)
    # ファイルに保存
    vis.save(timesteps, initial_placement, "output.txt")

出力例:
    TS 0 [2P] SplitOp, SplitOp
    T(0): []        M(1): [8,7,6,5,4]  T(2): [3,2,1,0]  L(3): []        T(4): [9]       M(5): [10,11,..]  T(6): []
                                         ^^^Split                          ^^^Split
"""

from typing import List, Dict, Optional, TextIO
import sys

from ast_def import (
    Operation, GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp,
    MeasureOp, ResetOp, TimeStep
)
from qccd_compiler import QCCDArchitecture
from validator import _ValidatorState


def _op_short_name(op: Operation) -> str:
    """操作の短縮名"""
    if isinstance(op, SplitOp):
        return f"Split(ion{op.ion_id}) z{op.src_zone_id}->z{op.dst_zone_id}"
    if isinstance(op, MergeOp):
        return f"Merge z{op.zone_left}+z{op.zone_right}"
    if isinstance(op, ShuttleOp):
        dst = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
        return f"Shuttle(ion{op.target}) z{op.src_zone}->z{dst}"
    if isinstance(op, SwapOp):
        return f"Swap({op.targets[0]},{op.targets[1]})"
    if isinstance(op, GateOp):
        if op.type == 'MS':
            return f"MS({op.targets[0]},{op.targets[1]})"
        return f"R({op.targets[0]})"
    if isinstance(op, MeasureOp):
        return f"Measure({op.targets})"
    if isinstance(op, ResetOp):
        return f"Reset({op.targets})"
    return str(type(op).__name__)


def _zone_type_label(arch: QCCDArchitecture, zone_id: int) -> str:
    """ゾーンのタイプラベル"""
    zone = arch.get_zone(zone_id)
    if zone is None:
        return "?"
    t = zone.zone_type
    if t in ("LIZ", "L"):
        return "L"
    elif t in ("Memory", "M"):
        return "M"
    else:
        return "T"


def _format_zone_contents(ions: List[int], max_width: int = 14) -> str:
    """ゾーン内のイオンリストを整形"""
    if not ions:
        return "[]"
    ion_strs = [str(i) for i in ions]
    full = "[" + ",".join(ion_strs) + "]"
    if len(full) <= max_width:
        return full
    # 省略表示
    shown = []
    current_len = 2  # "[" and "]"
    for s in ion_strs:
        needed = len(s) + (1 if shown else 0)  # comma
        if current_len + needed + 2 > max_width:  # +2 for ".."
            shown.append("..")
            break
        shown.append(s)
        current_len += needed
    return "[" + ",".join(shown) + "]"


class ZoneVisualizer:
    """TimeStep ごとのゾーン状態をテキストで可視化"""

    def __init__(self, architecture: QCCDArchitecture):
        self.arch = architecture
        self.zone_ids = sorted(z.id for z in architecture.zones)

    def _build_state(self, initial_placement: Dict[int, int]) -> _ValidatorState:
        """初期状態を構築"""
        state = _ValidatorState(self.arch)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion(ion_id, zone_id)
        return state

    def _apply_op(self, op: Operation, state: _ValidatorState):
        """1つのopを状態に適用"""
        if isinstance(op, SplitOp):
            state.apply_split(op.ion_id, op.dst_zone_id)
        elif isinstance(op, MergeOp):
            state.apply_merge(op.zone_left, op.zone_right)
        elif isinstance(op, ShuttleOp):
            dst = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
            state.apply_shuttle(op.target, dst)
        elif isinstance(op, SwapOp):
            state.apply_swap(op.targets[0], op.targets[1])
        # Gate/Measure/Reset は状態を変えない

    def _format_state_line(self, state: _ValidatorState, changed_zones: set = None) -> str:
        """ゾーン状態を1行で表示"""
        parts = []
        for zid in self.zone_ids:
            label = _zone_type_label(self.arch, zid)
            contents = state.zone_contents.get(zid, [])
            contents_str = _format_zone_contents(contents)
            marker = "*" if changed_zones and zid in changed_zones else " "
            parts.append(f"{marker}{label}({zid}):{contents_str}")
        return "  ".join(parts)

    def _get_changed_zones(self, before: Dict[int, List[int]], after: Dict[int, List[int]]) -> set:
        """状態変化があったゾーンを返す"""
        changed = set()
        for zid in self.zone_ids:
            b = before.get(zid, [])
            a = after.get(zid, [])
            if b != a:
                changed.add(zid)
        return changed

    def render_timesteps(
        self,
        timesteps: List[TimeStep],
        initial_placement: Dict[int, int],
        start: int = 0,
        end: Optional[int] = None,
        show_state: bool = True,
        show_ops: bool = True,
    ) -> List[str]:
        """
        TimeStep 列をテキスト行のリストとして返す

        Args:
            timesteps: TimeStep列
            initial_placement: 初期配置
            start: 表示開始TimeStep
            end: 表示終了TimeStep (exclusive, None=全部)
            show_state: ゾーン状態を表示するか
            show_ops: 操作内容を表示するか
        """
        state = self._build_state(initial_placement)
        lines = []

        if end is None:
            end = len(timesteps)

        # 初期状態を表示
        if start == 0:
            lines.append("=== Initial State ===")
            lines.append(self._format_state_line(state))
            lines.append("")

        # start まで早送り
        for ts_idx in range(min(start, len(timesteps))):
            for op in timesteps[ts_idx].ops:
                self._apply_op(op, state)

        # 表示範囲を処理
        for ts_idx in range(start, min(end, len(timesteps))):
            ts = timesteps[ts_idx]
            ops = ts.ops

            # ヘッダ
            par_label = f"[{len(ops)}P]" if len(ops) > 1 else ""
            op_names = ", ".join(_op_short_name(op) for op in ops)

            if show_ops:
                lines.append(f"--- TS {ts_idx} {par_label} {op_names} ---")

            # 状態変化前のスナップショット
            before = state.snapshot()

            # 全 ops を適用
            for op in ops:
                self._apply_op(op, state)

            # 状態変化後
            if show_state:
                changed = self._get_changed_zones(before, state.snapshot())
                lines.append(self._format_state_line(state, changed))

            # 並列ops がある場合は空行で区切り
            if len(ops) > 1:
                lines.append("")

        return lines

    def print_all(
        self,
        timesteps: List[TimeStep],
        initial_placement: Dict[int, int],
        file: TextIO = None,
    ):
        """全 TimeStep を出力"""
        if file is None:
            file = sys.stdout
        lines = self.render_timesteps(timesteps, initial_placement)
        for line in lines:
            print(line, file=file)

    def print_range(
        self,
        timesteps: List[TimeStep],
        initial_placement: Dict[int, int],
        start: int = 0,
        end: Optional[int] = None,
        file: TextIO = None,
    ):
        """指定範囲の TimeStep を出力"""
        if file is None:
            file = sys.stdout
        lines = self.render_timesteps(timesteps, initial_placement, start=start, end=end)
        for line in lines:
            print(line, file=file)

    def save(
        self,
        timesteps: List[TimeStep],
        initial_placement: Dict[int, int],
        filepath: str,
        start: int = 0,
        end: Optional[int] = None,
    ):
        """ファイルに保存"""
        lines = self.render_timesteps(timesteps, initial_placement, start=start, end=end)
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
