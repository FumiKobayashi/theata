"""Build the examples/ directory snapshot.

Regenerates all artifacts under examples/:
  - single_liz/dX_reuse1/        : 1-round compile + recompile + noise
  - multi_round/dX_reuse1_3rounds/: 3-round full pipeline + MWPM
  - distance_scan/                : V-shape plot/JSON (copied from results/)
  - README.md                     : overall guide

This is a snapshot regenerator, not a versioned archive. Each run overwrites.

Usage:
    python experiments/exp_build_examples.py
    python experiments/exp_build_examples.py --shots 50000   # MWPM shots for multi_round
"""

import sys
import argparse
import json
import math
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np
import stim
import pymatching

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from formatter import save_compile_artifacts
from pauli_lookup import PauliErrorLookup
from stim_recompiler import recompile_to_stim, recompile_multi_round
from exp_noise_injection import (
    phonon_to_error_rates,
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_transport_dephasing,
    inject_per_op_transport_dephasing,
    _per_round_dephasing_time_us,
    default_noise_params,
)
from exp_multi_round import placement_greedy_alt


# どの placement strategy を使うかの canonical 設定。
# 2026-04 のポスター期と同じ greedy_alt (回路の登場順に LIZ 近接 zone から左右交互配置)。
# 過去の split_even は examples/_archive/20260620_split_even/ に保存。
PLACEMENT_STRATEGY = "greedy_alt"

# d=7 以上は MS gate の transversal phonon が lookup table 上限 (n_bar=20) を多数
# 超過するため、 noisy.stim は出力しない (出力しても大半が clip artifact になる)。
# compile artifacts (flat_ops/timesteps/result.json) はそのまま生成して phonon
# scaling の plot に使う。
MAX_D_FOR_NOISE = 5


# Pauli-channel lookup table (transversal phonon → 15 Pauli probs, strict mode)
PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"


REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / 'examples'

# (d, forward_module, reverse_module, observable_qubits) — Rotated surface code 系
DISTANCE_CONFIG = {
    3: ('snake_reuse1', 'snake_reuse1_reverse', [0, 1, 2]),
    5: ('d5_reuse1', 'd5_reuse1_reverse_cnot', [0, 1, 2, 3, 4]),
    7: ('d7_reuse1', 'd7_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6]),
}

# Repetition code 系 (bit-flip protection、 Z-stab のみ)
# (d, module, observable_qubit) — 1 ラウンド only なので reverse 不要
REP_CODE_CONFIG = {
    3: ('d3_rep_reuse1', [0]),  # logical Z = Z_0 (任意の単 qubit Z で OK)
}


# Note: flat_ops / timesteps の formatter は src/formatter.py に統一済み。
# save_flat_ops_txt / save_parallel_txt をそちらから import して使う。


# ==========================================
# Compile helpers
# ==========================================

def make_arch(mod):
    half = (len(mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(mod.ANCILLA_QUBITS) + 4)
    return QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)


def initial_placement(mod):
    """Canonical placement: greedy_alt (回路の登場順に LIZ 近接 zone から左右交互配置)。

    過去の split_even (前半 zone 1, 後半 zone 5) 版は examples/_archive/20260620_split_even/
    に退避済み。 d=3 ベースで par_time が ~30 ms → 27 ms に改善する。
    """
    return placement_greedy_alt(mod.DATA_QUBITS, mod.ANCILLA_QUBITS, mod)


def compile_one(mod_name, placement, arch, initial_zone_contents=None):
    mod = __import__(mod_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    if initial_zone_contents is not None:
        ordered = {}
        for zid in sorted(initial_zone_contents.keys()):
            for ion_id in initial_zone_contents[zid]:
                ordered[ion_id] = zid
        placement = ordered
    compiler = QCCDCompiler(arch, config=CompilerConfig())
    return mod, compiler.compile(dag, placement)


def get_phonon_error_rates(results):
    """旧 DEPOLARIZE2 用 — placeholder p_MS のスカラー。"""
    all_gates = []
    for r in results:
        pd = r.phonon_data or {}
        all_gates.extend(pd.get('at_gates', []))
    return phonon_to_error_rates(all_gates)


def get_pauli_lookup_rates(results, mode=LOOKUP_MODE):
    """新 PAULI_CHANNEL_2 用 — 各 MS で 15 個の Pauli 確率を lookup table から取得。

    Returns:
        [(gate_name, [15 floats], n_bar_used, max_axial, max_transversal), ...]
    """
    all_gates = []
    for r in results:
        pd = r.phonon_data or {}
        all_gates.extend(pd.get('at_gates', []))
    return phonon_to_pauli_lookup_rates(all_gates, PAULI_LOOKUP, mode=mode)


# ==========================================
# Single-LIZ 1-round examples
# ==========================================

def build_single_liz(d: int, outdir: Path):
    fwd_name, _, _ = DISTANCE_CONFIG[d]
    print(f"  [single_liz d={d}] {fwd_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    fwd = __import__(fwd_name)
    arch = make_arch(fwd)
    placement = initial_placement(fwd)
    _, result = compile_one(fwd_name, placement, arch)

    # Dump all canonical compile artifacts via src/formatter.py
    save_compile_artifacts(
        result, str(outdir),
        initial_placement=placement, arch=arch,
        circuit_name=fwd_name, label=f"d{d}_reuse1",
        placement_strategy=PLACEMENT_STRATEGY,
        stabilizer_parallel=False,
    )

    # d=7 以上は MS の transversal phonon が lookup table 上限を多数超えるため、
    # noisy.stim を出力しても clip artifact だらけになる。 compile artifacts のみ
    # 残して noisy.stim は skip。
    if d > MAX_D_FOR_NOISE:
        print(f"    [skip noisy.stim for d={d}] transversal phonon exceeds lookup max")
        return

    # Recompiled (no detectors, no observable — simplest 1-round logical)
    circuit = recompile_to_stim(result)
    (outdir / "recompiled.stim").write_text(str(circuit), encoding='utf-8')

    # Noise injection: PAULI_CHANNEL_2 lookup (strict mode) for MS,
    # idle dephasing for everything else (current default approach B).
    pauli_rates = get_pauli_lookup_rates([result])
    noise_params = default_noise_params()

    noisy_base = inject_noise_approach_b(circuit, pauli_rates, noise_params)
    noisy_lumped = inject_transport_dephasing(noisy_base, [result], noise_params)
    noisy_perop = inject_per_op_transport_dephasing(noisy_base, [result], noise_params)
    (outdir / "noisy_lumped.stim").write_text(str(noisy_lumped), encoding='utf-8')
    (outdir / "noisy_perop.stim").write_text(str(noisy_perop), encoding='utf-8')

    # MS noise (PAULI_CHANNEL_2 lookup) の per-gate metadata を保存
    pauli_meta = {
        "lookup_mode": LOOKUP_MODE,
        "lookup_csv": str(PAULI_LOOKUP.csv_path.name),
        "n_bar_range": list(PAULI_LOOKUP.n_bar_range),
        "per_ms": [
            {
                "gate": r[0],
                "n_bar_used": r[2],
                "max_axial": r[3],
                "max_transversal": r[4],
                "sum_error_prob": sum(r[1]),
            }
            for r in pauli_rates
        ],
    }
    (outdir / "ms_noise.json").write_text(
        json.dumps(pauli_meta, indent=2), encoding='utf-8')


# ==========================================
# Multi-round full pipeline
# ==========================================

def build_multi_round(d: int, rounds: int, shots: int, outdir: Path, seed: int = 42):
    fwd_name, rev_name, observable = DISTANCE_CONFIG[d]
    print(f"  [multi_round d={d}, {rounds} rounds] {fwd_name}/{rev_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    fwd = __import__(fwd_name)
    arch = make_arch(fwd)
    placement = initial_placement(fwd)

    results = []
    cur_p = placement
    cur_zc = None
    for r in range(1, rounds + 1):
        mod_name = fwd_name if r % 2 == 1 else rev_name
        _, result = compile_one(mod_name, cur_p, arch, initial_zone_contents=cur_zc)
        results.append(result)
        cur_p = result.final_placement
        cur_zc = result.final_zone_contents

    # Recompiled noiseless (DETECTOR + OBSERVABLE)
    circuit = recompile_multi_round(results, observable_column=observable)
    (outdir / "recompiled.stim").write_text(str(circuit), encoding='utf-8')

    # Noise injection: PAULI_CHANNEL_2 lookup (strict) for MS, idle dephasing elsewhere
    pauli_rates = get_pauli_lookup_rates(results)
    noise_params = default_noise_params()
    noisy_base = inject_noise_approach_b(circuit, pauli_rates, noise_params)
    noisy_lumped = inject_transport_dephasing(noisy_base, results, noise_params)
    noisy_perop = inject_per_op_transport_dephasing(noisy_base, results, noise_params)
    (outdir / "noisy_lumped.stim").write_text(str(noisy_lumped), encoding='utf-8')
    (outdir / "noisy_perop.stim").write_text(str(noisy_perop), encoding='utf-8')

    # MWPM on lumped version (per-op is mathematically near-identical for current shots)
    # `approximate_disjoint_errors=True` は PAULI_CHANNEL_2 を含む回路で必須
    # (15 個の Pauli 確率を disjoint と近似する)
    sampler = noisy_lumped.compile_detector_sampler(seed=seed)
    dem = noisy_lumped.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det_samples, obs_actual = sampler.sample(shots, separate_observables=True)
    obs_predicted = matcher.decode_batch(det_samples)
    errors = int(np.any(obs_predicted != obs_actual, axis=1).sum())
    p_L = errors / shots
    stderr = math.sqrt(p_L * (1 - p_L) / shots) if shots > 0 else 0.0
    per_round = 1 - (1 - p_L) ** (1.0 / rounds) if p_L > 0 else 0.0

    mwpm_result = {
        "distance": d,
        "rounds": rounds,
        "shots": shots,
        "seed": seed,
        "transport_mode": "lumped",
        "ms_noise_model": f"PAULI_CHANNEL_2 (lookup, {LOOKUP_MODE})",
        "errors": errors,
        "p_L": p_L,
        "stderr_1sigma": stderr,
        "per_round_p_L_approx": per_round,
        "circuit_files": {
            "noiseless": "recompiled.stim",
            "noisy_lumped": "noisy_lumped.stim",
            "noisy_perop": "noisy_perop.stim",
        },
    }
    (outdir / "mwpm_result.json").write_text(
        json.dumps(mwpm_result, indent=2), encoding='utf-8')

    # Wait time per round
    wait_us = [_per_round_dephasing_time_us(cr) for cr in results]
    (outdir / "round_stats.json").write_text(
        json.dumps({
            "rounds": rounds,
            "wait_us_per_round": wait_us,
            "wait_us_total": sum(wait_us),
        }, indent=2),
        encoding='utf-8'
    )
    print(f"    p_L = {p_L:.4e} +/- {stderr:.4e}  (errors={errors}/{shots})")


# ==========================================
# Repetition code (single LIZ, 1 round, full pipeline incl. DETECTOR/OBSERVABLE)
# ==========================================

def build_rep_code(d: int, outdir: Path, shots: int = 10000, seed: int = 42):
    """d=3 repetition code を 1 ラウンド full pipeline で出力 (スライド用)。

    生成物:
      - flat_ops.txt, timesteps.txt, result.json (compile artifacts)
      - recompiled.stim  : 1 ラウンドの clean logical Stim (DETECTOR + OBSERVABLE_INCLUDE)
        — 便宜的に "入力 Stim" のスタンドインとして使う
      - noisy_lumped.stim, noisy_perop.stim : ノイズ注入版
      - ms_noise.json : 各 MS の n_bar / 15 probs sum
      - mwpm_result.json : 1 ラウンドだけだと意味は薄いが構造確認のため
    """
    if d not in REP_CODE_CONFIG:
        print(f"  [skip rep d={d}] not configured")
        return
    mod_name, observable = REP_CODE_CONFIG[d]
    print(f"  [rep_code d={d}, 1 round] {mod_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    mod = __import__(mod_name)
    arch = make_arch(mod)
    placement = initial_placement(mod)
    _, result = compile_one(mod_name, placement, arch)

    # compile artifacts
    save_compile_artifacts(
        result, str(outdir),
        initial_placement=placement, arch=arch,
        circuit_name=mod_name, label=f"d{d}_rep_reuse1",
        placement_strategy=PLACEMENT_STRATEGY,
        stabilizer_parallel=False,
    )

    # recompile 1-round with DETECTOR/OBSERVABLE
    circuit = recompile_multi_round([result], observable_column=observable)
    (outdir / "recompiled.stim").write_text(str(circuit), encoding='utf-8')

    # noise injection (PAULI_CHANNEL_2 strict)
    pauli_rates = get_pauli_lookup_rates([result])
    noise_params = default_noise_params()
    noisy_base = inject_noise_approach_b(circuit, pauli_rates, noise_params)
    noisy_lumped = inject_transport_dephasing(noisy_base, [result], noise_params)
    noisy_perop = inject_per_op_transport_dephasing(noisy_base, [result], noise_params)
    (outdir / "noisy_lumped.stim").write_text(str(noisy_lumped), encoding='utf-8')
    (outdir / "noisy_perop.stim").write_text(str(noisy_perop), encoding='utf-8')

    # MS noise per-gate metadata
    pauli_meta = {
        "lookup_mode": LOOKUP_MODE,
        "lookup_csv": str(PAULI_LOOKUP.csv_path.name),
        "n_bar_range": list(PAULI_LOOKUP.n_bar_range),
        "per_ms": [
            {
                "gate": r[0],
                "n_bar_used": r[2],
                "max_axial": r[3],
                "max_transversal": r[4],
                "sum_error_prob": sum(r[1]),
            }
            for r in pauli_rates
        ],
    }
    (outdir / "ms_noise.json").write_text(
        json.dumps(pauli_meta, indent=2), encoding='utf-8')

    # MWPM (1 round の意味は薄いが、構造確認用)
    try:
        sampler = noisy_lumped.compile_detector_sampler(seed=seed)
        dem = noisy_lumped.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        det_samples, obs_actual = sampler.sample(shots, separate_observables=True)
        obs_predicted = matcher.decode_batch(det_samples)
        errors = int(np.any(obs_predicted != obs_actual, axis=1).sum())
        p_L = errors / shots
        stderr = math.sqrt(p_L * (1 - p_L) / shots) if shots > 0 else 0.0
        mwpm_result = {
            "distance": d,
            "code": "repetition (bit-flip)",
            "rounds": 1,
            "shots": shots,
            "seed": seed,
            "transport_mode": "lumped",
            "ms_noise_model": f"PAULI_CHANNEL_2 (lookup, {LOOKUP_MODE})",
            "observable": observable,
            "errors": errors,
            "p_L": p_L,
            "stderr_1sigma": stderr,
        }
        (outdir / "mwpm_result.json").write_text(
            json.dumps(mwpm_result, indent=2), encoding='utf-8')
        print(f"    p_L = {p_L:.4e} +/- {stderr:.4e}  (errors={errors}/{shots})")
    except Exception as e:
        print(f"    [WARN] MWPM skipped: {type(e).__name__}: {e}")


# ==========================================
# Distance scan: copy latest results
# ==========================================

def build_distance_scan(outdir: Path):
    """Find the most recent distance_scan_*.{png,json} in results/noise_injection/
    and copy them into examples/distance_scan/ with stable names.
    """
    src_dir = REPO_ROOT / 'results' / 'noise_injection'
    pngs = sorted(src_dir.glob('distance_scan_*.png'),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    jsons = sorted(src_dir.glob('distance_scan_*.json'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not pngs or not jsons:
        print(f"  [distance_scan] no source files found in {src_dir}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pngs[0], outdir / 'plot.png')
    shutil.copy(jsons[0], outdir / 'data.json')
    print(f"  [distance_scan] copied {pngs[0].name} and {jsons[0].name}")

    # Write a small README explaining the finding
    readme = """# Distance scan: d=3 / d=5 (single-LIZ, PAULI_CHANNEL_2 lookup, strict)

Logical error rate p_L as a function of code distance d for the snake_reuseN
family (single-LIZ compiler, 3 rounds of QEC, MWPM decoding with pymatching).

## Files
- `plot.png` — p_L vs d with 1-sigma error bars (log scale on y)
- `data.json` — raw numbers (errors, shots, p_L, stderr)

## Noise model (current)

- **MS gates**: PAULI_CHANNEL_2 with 15 Pauli probabilities looked up from
  `src/pauli_twirled_error_probabilities.csv`. Per MS, the transversal phonon
  is taken as `n_bar`, rounded up to the nearest table key (**strict mode**),
  and the corresponding 15 probabilities are emitted.
- **R / RESET / MEASURE**: scalar probabilities (p_R=1e-4, p_reset=8e-4,
  p_meas=3.2e-3). idle dephasing during these ops applies Z_ERROR on
  non-target qubits.
- **Transport**: lumped Z_ERROR per round on all qubits, with duration =
  `wall_clock − sum(gate_active)` (= timestep critical paths minus active
  gate time), Γ_φ = 3.0 /s.

## Key finding: monotonic degradation with d (single-LIZ)

| d | p_L (3 rounds, 50k shots) |
|---|---------------------------|
| 3 | 1.87e-2 ± 6e-4            |
| 5 | 3.64e-1 ± 2e-3            |

p_L is **strictly increasing** with d in the single-LIZ compiler — surface
code protection is overwhelmed by accumulated MS + transport noise that
itself scales (worse than the code's d-scaling).

Compared to a previous run with placeholder MS rates (where a small
non-monotonic V-shape appeared: d=3→5 improved, then 5→7 degraded), the
properly-calibrated noise pushes single-LIZ d=5 well past the point of
useful protection.

## Interpretation

Larger d brings:
- More qubits to shuttle → longer transport time per round
- More stabilizers per round → more MS gates exposed to motional heating
- Higher transversal phonon at later MS gates → higher per-MS error
  (n_bar grows from ~2 at d=3 to ~11 average at d=5, and 20+ at the worst
  MS in d=5 — already clipping the lookup table)

This is the strongest empirical justification for the multi-LIZ direction:
parallelizing transport across LIZs should reduce both transport time *and*
per-MS phonon accumulation, restoring d-scaling of p_L.

## Caveats
- Strict mode (ceil on n_bar) is conservative; round_off / optimistic would
  give slightly lower numbers. The trend (monotonic increase) is unchanged.
- d=5 has 1 MS gate with transversal phonon slightly above the table max
  (20.69 vs table max 20.0); clipped to 20.0 with a runtime warning.
- d=7 is not included: many MS gates exceed the lookup table range, making
  comparison less meaningful with current strict-clipping behavior.
"""
    (outdir / 'README.md').write_text(readme, encoding='utf-8')


# ==========================================
# Overall README
# ==========================================

def write_top_readme():
    readme = """# examples/

Snapshot artifacts from the QCCD compiler + Stim recompile + noise injection
pipeline. **Regenerated by `experiments/exp_build_examples.py`** — not a
versioned archive. Old files are overwritten when new results come in.

## Layout

```
examples/
├── single_liz/                # 1-round compile + recompile + noise (no MWPM)
│   ├── d3_reuse1/             # snake_reuse1 (d=3)
│   ├── d5_reuse1/             # d5_reuse1
│   └── d7_reuse1/             # d7_reuse1
├── multi_round/               # 3-round full pipeline (DETECTOR/OBSERVABLE/MWPM)
│   ├── d3_reuse1_3rounds/
│   └── d5_reuse1_3rounds/
└── distance_scan/             # V-shape finding (d=3/5/7 with 3 rounds)
    ├── plot.png
    ├── data.json
    └── README.md
```

## Per-directory file conventions

### single_liz/dX_reuse1/

| file | content |
|------|---------|
| `flat_ops.txt` | Native instruction stream from the QCCD compiler (RESET / R / MS / MEASURE / Shuttle / Split / Merge / Swap), one op per line |
| `timesteps.txt` | Scheduled TimeStep stream with critical-path durations and cumulative wall-clock |
| `result.json` | Canonical compile result JSON (circuit / architecture / placement / op stats / timing / phonon tracking, bundled). Format matches `results/20260330TimeWithPhonon/*.json`. |
| `recompiled.stim` | Stim logical circuit (R / H / CX / M, with TICK between every op) — no detectors, no observable, no noise |
| `noisy_lumped.stim` | `recompiled.stim` + Approach-B per-op gate noise + lumped transport-time Z_ERROR (one Z_ERROR per round at start) |
| `noisy_perop.stim` | Same as above but transport-time Z_ERROR is split per anchor (R/CX/M boundary) |

### multi_round/dX_reuse1_3rounds/

| file | content |
|------|---------|
| `recompiled.stim` | 3-round logical Stim circuit with cross-round DETECTOR (identity match on `(basis, frozenset(data_qubits))`) + final OBSERVABLE_INCLUDE |
| `noisy_lumped.stim` | Above + noise (transport lumped) |
| `noisy_perop.stim` | Above + noise (transport per-op) |
| `mwpm_result.json` | MWPM decoding result: errors, shots, p_L, 1-sigma stderr |
| `round_stats.json` | Per-round wait time decomposition |

### distance_scan/

V-shape plot showing the limitation of the single-LIZ compiler (see directory's own README).

## How to regenerate

```bash
python experiments/exp_build_examples.py             # default 10000 shots
python experiments/exp_build_examples.py --shots 100000
```

## What's NOT here (planned)

- `multi_liz/`: per-core stream output (waiting on Phase B of the per-LIZ stream redesign — see `memory/project_per_liz_stream_design.md`)
"""
    (EXAMPLES_DIR / 'README.md').write_text(readme, encoding='utf-8')


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Build examples/ snapshot')
    parser.add_argument('--distances', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--rounds', type=int, default=3,
                        help='Rounds for multi_round/ artifact')
    parser.add_argument('--shots', type=int, default=10000,
                        help='MWPM shots for multi_round/ artifact')
    parser.add_argument('--skip-distance-scan', action='store_true')
    args = parser.parse_args()

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {EXAMPLES_DIR}")

    # single_liz/
    print("\n[single_liz/]")
    for d in args.distances:
        if d not in DISTANCE_CONFIG:
            print(f"  [skip] d={d} not configured")
            continue
        build_single_liz(d, EXAMPLES_DIR / 'single_liz' / f'd{d}_reuse1')

    # multi_round/ for d=3 and d=5 (distance_scan で取れる範囲。d=7 は phonon が
    # lookup table 範囲を多数超えるためここでは出力しない)
    print(f"\n[multi_round/] (d=3 and d=5, {args.rounds} rounds, {args.shots} shots)")
    for d in (3, 5):
        if d in args.distances and d in DISTANCE_CONFIG:
            build_multi_round(
                d, args.rounds, args.shots,
                EXAMPLES_DIR / 'multi_round' / f'd{d}_reuse1_{args.rounds}rounds',
            )

    # rep_code/ — d=3 repetition code, 1 round only (スライド用)
    print(f"\n[rep_code/] (d=3 repetition code, 1 round, {args.shots} shots)")
    for d in REP_CODE_CONFIG:
        build_rep_code(d, EXAMPLES_DIR / 'rep_code' / f'd{d}_rep_reuse1',
                       shots=args.shots)

    # distance_scan/
    if not args.skip_distance_scan:
        print("\n[distance_scan/]")
        build_distance_scan(EXAMPLES_DIR / 'distance_scan')

    # Top README
    write_top_readme()
    print(f"\n[README.md] written")
    print(f"\nDone. Output: {EXAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
