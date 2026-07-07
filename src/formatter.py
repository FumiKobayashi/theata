"""Compiler output formatters — human-readable dumps of native streams.

公開関数:
  - format_op_summary(op)          : 単一 Operation の短い要約文字列
  - format_flat_ops_text(...)      : flat_ops の全体テキスト (header + 各 op を 1 行)
  - save_flat_ops_txt(...)         : 上記をファイル保存
  - format_timesteps_text(...)     : TimeStep stream のテキスト (re-export)
  - save_parallel_txt(...)         : TimeStep stream をファイル保存 (re-export)

Why this module exists:
  各実験 script が独自に dump 関数を書いてフォーマットが揺れていた問題への対処。
  コンパイラの出力は一律なので、その人間可読表現も canonical な 1 箇所にまとめる。

Note:
  format_timesteps_text と save_parallel_txt は元々 parallel_estimator.py に
  あり、後方互換性のためあちら側にも残してある。新規コードはこのモジュールから
  import することを推奨。
"""

from pathlib import Path
from typing import List, Optional, Dict

from ast_def import (
    Operation, GateOp, ResetOp, MeasureOp,
    ShuttleOp, SwapOp, SplitOp, MergeOp, BarrierOp, CoolOp, TimeStep,
)


# ==========================================
# 単一 op の要約
# ==========================================

def format_op_summary(op: Operation) -> str:
    """単一 Operation の短い要約文字列。

    parallel_estimator.ParallelEstimator._op_summary と同一の出力規約。
    """
    if isinstance(op, GateOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"{op.type}({ions}) z{op.zone_id}"
    if isinstance(op, ResetOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Reset({ions}) z{op.zone_id}"
    if isinstance(op, MeasureOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Measure({ions}) z{op.zone_id}"
    if isinstance(op, SplitOp):
        return f"Split(ion{op.ion_id}) z{op.src_zone_id}->z{op.dst_zone_id}"
    if isinstance(op, MergeOp):
        return f"Merge z{op.zone_left}+z{op.zone_right}"
    if isinstance(op, SwapOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Swap({ions}) z{op.zone_id}"
    if isinstance(op, ShuttleOp):
        return f"Shuttle(ion{op.target}) z{op.src_zone}->{op.dst}"
    if isinstance(op, BarrierOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Barrier({ions})"
    if isinstance(op, CoolOp):
        coolant = f" coolant={op.coolant_id}" if op.coolant_id >= 0 else ""
        return f"Cool z{op.zone_id}{coolant}"
    return str(op)


# ==========================================
# flat_ops formatter (新規)
# ==========================================

def format_flat_ops_text(
    flat_ops: List[Operation],
    initial_placement: Optional[Dict[int, int]] = None,
    arch=None,
) -> str:
    """Native instruction stream (flat_ops) を人間可読テキストにする。

    出力形式は format_timesteps_text のヘッダー方針と整合させる:
        # Architecture: z0:T(20) - z1:M(20) - ...
        # Initial placement: {ion: zone, ...}
        # Total ops: <N>
        # Op breakdown: Split=120, Merge=120, Shuttle=20, ...

           0  Split(ion7) z9->z4
           1  Swap(8,9) z9
           2  Merge z4+z5
           ...
    """
    lines: List[str] = []

    # ヘッダー (timesteps formatter と同じ流儀)
    if arch is not None:
        layout = " - ".join(f"z{z.id}:{z.zone_type}({z.capacity})" for z in arch.zones)
        lines.append(f"# Architecture: {layout}")
    if initial_placement is not None:
        lines.append(f"# Initial placement: {initial_placement}")

    # op-type breakdown
    breakdown: Dict[str, int] = {}
    for op in flat_ops:
        key = _op_type_key(op)
        breakdown[key] = breakdown.get(key, 0) + 1
    breakdown_str = ", ".join(f"{k}={v}" for k, v in sorted(breakdown.items()))
    lines.append(f"# Total ops: {len(flat_ops)}")
    lines.append(f"# Op breakdown: {breakdown_str}")
    lines.append("")

    # 各 op を 1 行
    idx_width = len(str(len(flat_ops) - 1)) if flat_ops else 1
    for i, op in enumerate(flat_ops):
        lines.append(f"{i:>{idx_width}}  {format_op_summary(op)}")

    return "\n".join(lines)


def _op_type_key(op: Operation) -> str:
    if isinstance(op, GateOp):
        return op.type  # "MS" or "R"
    if isinstance(op, ResetOp):
        return "Reset"
    if isinstance(op, MeasureOp):
        return "Measure"
    if isinstance(op, ShuttleOp):
        return "Shuttle"
    if isinstance(op, SwapOp):
        return "Swap"
    if isinstance(op, SplitOp):
        return "Split"
    if isinstance(op, MergeOp):
        return "Merge"
    if isinstance(op, BarrierOp):
        return "Barrier"
    if isinstance(op, CoolOp):
        return "Cool"
    return type(op).__name__


def save_flat_ops_txt(
    flat_ops: List[Operation],
    filepath: str,
    initial_placement: Optional[Dict[int, int]] = None,
    arch=None,
) -> None:
    """flat_ops をテキストファイルに保存。"""
    text = format_flat_ops_text(flat_ops, initial_placement=initial_placement, arch=arch)
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n", encoding='utf-8')


# ==========================================
# timesteps formatter (re-export)
# ==========================================
# 元の実装は parallel_estimator.py にある。新規コードはこちらから import 推奨。

from parallel_estimator import (  # noqa: E402
    ParallelEstimator,
    format_timesteps_text,
    save_parallel_txt,
)


# ==========================================
# Canonical compile result JSON (compile meta + phonon, bundled)
# ==========================================
# 既存の `results/20260330TimeWithPhonon/*.json` 形式に合わせる:
#   circuit, label, generated_at, architecture, initial_placement,
#   stats, total_ops, total_timesteps, seq_time_ms, par_time_ms, valid,
#   phonon { max_axial, max_transversal, per_ion_axial, per_ion_transversal, at_gates }

def _capacity_for(arch, zone_type: str) -> Optional[int]:
    """Return the (representative) capacity of zones with given type."""
    if arch is None:
        return None
    caps = [z.capacity for z in arch.zones if z.zone_type == zone_type]
    return caps[0] if caps else None


def build_compile_result_json(
    compile_result,
    initial_placement: Optional[Dict[int, int]] = None,
    arch=None,
    circuit_name: Optional[str] = None,
    label: Optional[str] = None,
    valid: Optional[bool] = None,
    placement_strategy: Optional[str] = None,
    stabilizer_parallel: Optional[bool] = None,
) -> dict:
    """canonical な compile result JSON dict を構築する。

    既存の `results/20260330TimeWithPhonon/*.json` 形式を踏襲。
    compile meta と phonon tracking を 1 つの dict に bundle する。
    """
    from datetime import datetime

    flat_ops = compile_result.flat_ops
    op_breakdown: Dict[str, int] = {}
    for op in flat_ops:
        k = _op_type_key(op)
        op_breakdown[k] = op_breakdown.get(k, 0) + 1

    est = ParallelEstimator()
    est_result = est.estimate(compile_result.timesteps)

    out: dict = {}
    if circuit_name is not None:
        out["circuit"] = circuit_name
    if label is not None:
        out["label"] = label
    out["generated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if arch is not None:
        out["architecture"] = {
            "liz_capacity": _capacity_for(arch, "L"),
            "memory_capacity": _capacity_for(arch, "M"),
            "temp_capacity": _capacity_for(arch, "T"),
            "layout": " - ".join(f"z{z.id}:{z.zone_type}({z.capacity})"
                                 for z in arch.zones),
        }

    if initial_placement is not None:
        out["initial_placement"] = {
            str(k): v for k, v in initial_placement.items()
        }

    # compile_config: CompilerConfig + placement_strategy (caller指定) を記録
    # この情報が無いと「同じコンパイラに同じ回路を入れたのに結果が違う」原因の追跡が
    # 出来なくなる (split_even vs greedy_alt の regression と同種の問題)。必ず保存する。
    config_dict: Dict = {}
    if placement_strategy is not None:
        config_dict["placement_strategy"] = placement_strategy
    if stabilizer_parallel is not None:
        config_dict["stabilizer_parallel"] = stabilizer_parallel
    cc = getattr(compile_result, 'config', None)
    if cc is not None and hasattr(cc, 'to_dict'):
        for k, v in cc.to_dict().items():
            config_dict[k] = v
    if config_dict:
        out["compile_config"] = config_dict

    # stats: 各 op type の count を上位フィールドに揃える ("gates", "shuttles" 等)
    stats = {
        "gates": op_breakdown.get("MS", 0) + op_breakdown.get("R", 0),
        "shuttles": op_breakdown.get("Shuttle", 0),
        "swaps": op_breakdown.get("Swap", 0),
        "splits": op_breakdown.get("Split", 0),
        "merges": op_breakdown.get("Merge", 0),
        "resets": op_breakdown.get("Reset", 0),
        "measures": op_breakdown.get("Measure", 0),
        "cools": op_breakdown.get("Cool", 0),
        "timesteps": len(compile_result.timesteps),
    }
    out["stats"] = stats
    out["op_breakdown"] = op_breakdown   # 各 op type の生 count (補足情報)
    out["total_ops"] = len(flat_ops)
    out["total_timesteps"] = len(compile_result.timesteps)
    out["seq_time_ms"] = est_result.sequential_time / 1000
    out["par_time_ms"] = est_result.total_time / 1000
    out["speedup"] = est_result.speedup
    out["avg_parallelism"] = est_result.parallelism

    if valid is not None:
        out["valid"] = valid

    # phonon (compile_result.phonon_data がある場合のみ)
    pd = getattr(compile_result, 'phonon_data', None)
    if pd:
        at_gates_serial = []
        for entry in pd.get('at_gates', []):
            phonons = entry.get('phonons', {})
            at_gates_serial.append({
                "gate": entry.get('gate'),
                "phonons": {str(ion): list(val) for ion, val in phonons.items()},
            })
        out["phonon"] = {
            "max_axial": pd.get('max_axial', 0.0),
            "max_transversal": pd.get('max_transversal', 0.0),
            "per_ion_axial": {
                str(ion): val for ion, val in pd.get('per_ion_axial', {}).items()
            },
            "per_ion_transversal": {
                str(ion): val for ion, val in pd.get('per_ion_transversal', {}).items()
            },
            "at_gates": at_gates_serial,
        }

    return out


def save_compile_result_json(
    compile_result,
    filepath: str,
    initial_placement: Optional[Dict[int, int]] = None,
    arch=None,
    circuit_name: Optional[str] = None,
    label: Optional[str] = None,
    valid: Optional[bool] = None,
    placement_strategy: Optional[str] = None,
    stabilizer_parallel: Optional[bool] = None,
) -> None:
    """canonical な compile result JSON をファイルに保存する。

    Stim 命令数や MWPM 結果のような「コンパイラの出力に直接含まれない」
    メタは別ファイル (例: noise_meta.json, mwpm_result.json) に分けること。
    """
    import json
    data = build_compile_result_json(
        compile_result,
        initial_placement=initial_placement,
        arch=arch,
        circuit_name=circuit_name,
        label=label,
        valid=valid,
        placement_strategy=placement_strategy,
        stabilizer_parallel=stabilizer_parallel,
    )
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                 encoding='utf-8')


# ==========================================
# One-call helper: 全 compile artifacts を一気に保存
# ==========================================

def save_compile_artifacts(
    compile_result,
    outdir: str,
    initial_placement: Optional[Dict[int, int]] = None,
    arch=None,
    circuit_name: Optional[str] = None,
    label: Optional[str] = None,
    valid: Optional[bool] = None,
    placement_strategy: Optional[str] = None,
    stabilizer_parallel: Optional[bool] = None,
) -> Dict[str, str]:
    """compile_result から canonical な全 artifact を outdir/ に保存する。

    保存ファイル:
      - flat_ops.txt   : native instruction stream
      - timesteps.txt  : scheduled timestep stream (parallel_estimator 形式)
      - result.json    : compile meta + phonon tracking (canonical JSON)

    Returns:
        書き出したファイル名 → 絶対パス の dict
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    save_flat_ops_txt(
        compile_result.flat_ops, str(out / "flat_ops.txt"),
        initial_placement=initial_placement, arch=arch,
    )
    save_parallel_txt(
        compile_result.timesteps, str(out / "timesteps.txt"),
        initial_placement=initial_placement, arch=arch,
    )
    save_compile_result_json(
        compile_result, str(out / "result.json"),
        initial_placement=initial_placement, arch=arch,
        circuit_name=circuit_name, label=label, valid=valid,
        placement_strategy=placement_strategy,
        stabilizer_parallel=stabilizer_parallel,
    )
    return {
        "flat_ops": str(out / "flat_ops.txt"),
        "timesteps": str(out / "timesteps.txt"),
        "result": str(out / "result.json"),
    }


__all__ = [
    "format_op_summary",
    "format_flat_ops_text",
    "save_flat_ops_txt",
    "format_timesteps_text",
    "save_parallel_txt",
    "build_compile_result_json",
    "save_compile_result_json",
    "save_compile_artifacts",
]
