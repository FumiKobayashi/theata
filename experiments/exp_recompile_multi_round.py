"""Multi-round recompile smoke test.

Goal: produce a multi-round Stim circuit (forward / reverse alternating) with
DETECTOR + OBSERVABLE_INCLUDE, and verify it's syntactically valid and that
the noiseless detector samples are all zero.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import stim

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from stim_recompiler import recompile_multi_round
from exp_noise_injection import (
    phonon_to_error_rates,
    inject_noise_approach_b,
    inject_transport_dephasing,
    inject_per_op_transport_dephasing,
    _per_round_dephasing_time_us,
    _per_anchor_wait_times_us,
    default_noise_params,
)


def _compile_round(circuit_name, initial_placement, arch, initial_zone_contents=None):
    """1 ラウンドを compile する。

    initial_zone_contents が与えられたらそれに従って initial_placement を再構築。
    """
    mod = __import__(circuit_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()

    if initial_zone_contents is not None:
        ordered = {}
        for zone_id in sorted(initial_zone_contents.keys()):
            for ion_id in initial_zone_contents[zone_id]:
                ordered[ion_id] = zone_id
        initial_placement = ordered

    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial_placement)
    return mod, result


def main():
    parser = argparse.ArgumentParser(description='Multi-round recompile smoke test')
    parser.add_argument('--forward', default='snake_reuse1')
    parser.add_argument('--reverse', default='snake_reuse1_reverse')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--observable', nargs='*', type=int, default=[0, 1, 2],
                        help='Data qubit IDs for logical Z observable; row chain connecting '
                             'left/right Z-boundaries (d=3 default: 0 1 2 = top row)')
    parser.add_argument('--dump', metavar='PATH',
                        help='Save Stim circuit to file')
    parser.add_argument('--shots', type=int, default=100,
                        help='Detector sample shots for noiseless verification')
    parser.add_argument('--noise', action='store_true',
                        help='Also produce a noisy version via inject_noise_approach_b')
    parser.add_argument('--noise-shots', type=int, default=1000,
                        help='Detector sample shots for noisy circuit (default: 1000)')
    parser.add_argument('--transport-mode', choices=['lumped', 'per-op', 'none'],
                        default='lumped',
                        help='Transport-time dephasing placement: '
                             '"lumped" (A: one Z_ERROR per round at start, math-equivalent, compact); '
                             '"per-op" (B: Z_ERROR before each anchor R/CX/M, visible per-gate timing); '
                             '"none" (skip transport dephasing entirely)')
    parser.add_argument('--dump-noisy', metavar='PATH',
                        help='Save noisy Stim circuit to file')
    args = parser.parse_args()

    # Load forward to know data/ancilla, set capacity
    fwd_mod = __import__(args.forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)

    initial = {}
    for i, q in enumerate(fwd_mod.DATA_QUBITS):
        initial[q] = 1 if i < len(fwd_mod.DATA_QUBITS) // 2 else 5
    for a in fwd_mod.ANCILLA_QUBITS:
        initial[a] = 5

    print(f"Forward: {args.forward}, Reverse: {args.reverse}, Rounds: {args.rounds}")
    print(f"  data={len(fwd_mod.DATA_QUBITS)}, ancillae={fwd_mod.ANCILLA_QUBITS}")

    results = []
    current_placement = initial
    current_zone_contents = None
    for r in range(1, args.rounds + 1):
        circuit_name = args.forward if r % 2 == 1 else args.reverse
        _, result = _compile_round(circuit_name, current_placement, arch,
                                   initial_zone_contents=current_zone_contents)
        print(f"  Round {r} ({circuit_name}): TS={len(result.timesteps)}, ops={len(result.flat_ops)}")
        results.append(result)
        current_placement = result.final_placement
        current_zone_contents = result.final_zone_contents

    print(f"\nRecompiling to multi-round Stim...")
    circuit = recompile_multi_round(results, observable_column=args.observable)

    # Count by instruction
    op_counts = {}
    for inst in circuit:
        op_counts[inst.name] = op_counts.get(inst.name, 0) + 1
    print(f"  num_qubits: {circuit.num_qubits}")
    print(f"  Total instructions: {sum(op_counts.values())}")
    print(f"  Op breakdown:")
    for name in sorted(op_counts):
        print(f"    {name}: {op_counts[name]}")

    # Verify detector sampling (noiseless → all zero)
    print(f"\nDetector sampling ({args.shots} shots, noiseless)...")
    try:
        sampler = circuit.compile_detector_sampler()
        det_samples, obs_samples = sampler.sample(args.shots, separate_observables=True)
        det_total = det_samples.sum()
        obs_total = obs_samples.sum()
        print(f"  det shape: {det_samples.shape}, num nonzero: {det_total}")
        print(f"  obs shape: {obs_samples.shape}, num nonzero: {obs_total}")
        if det_total == 0 and obs_total == 0:
            print(f"  [OK] All detectors and observables are 0 (noiseless circuit consistent)")
        else:
            print(f"  [WARN] Some detectors/observables are non-zero under noiseless sim")
    except Exception as e:
        print(f"  [FAIL] Detector sampler raised: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if args.dump:
        out_path = Path(args.dump)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(str(circuit))
        print(f"\n  Saved: {out_path}")

    if args.noise:
        print(f"\n{'='*60}")
        print(f"Phase 2: noise injection (Approach B, transport-time dephasing not yet)")
        print(f"{'='*60}")

        # 各ラウンドの phonon_at_gates を連結
        all_phonon_gates = []
        for r_idx, r in enumerate(results):
            pd = r.phonon_data or {}
            gates = pd.get('at_gates', [])
            all_phonon_gates.extend(gates)
        print(f"  Concatenated phonon entries across rounds: {len(all_phonon_gates)}")
        error_rates = phonon_to_error_rates(all_phonon_gates)
        print(f"  Error rates per MS: {len(error_rates)}")
        if error_rates:
            ms_rates = [r[1] for r in error_rates]
            print(f"    p_MS range: {min(ms_rates):.4e} -- {max(ms_rates):.4e}")

        noise_params = default_noise_params()
        noisy = inject_noise_approach_b(circuit, error_rates, noise_params)

        # Phase 2: transport-time dephasing
        transport_us = [_per_round_dephasing_time_us(cr) for cr in results]
        print(f"  Wait-time per round (us, includes pure transport + hidden idle): "
              f"{[f'{t:.1f}' for t in transport_us]}")

        if args.transport_mode == 'lumped':
            print(f"  Transport mode: LUMPED (A) -- one Z_ERROR per round at start")
            noisy = inject_transport_dephasing(noisy, results, noise_params)
        elif args.transport_mode == 'per-op':
            n_anchors = sum(len(_per_anchor_wait_times_us(cr)[0]) for cr in results)
            print(f"  Transport mode: PER-OP (B) -- one Z_ERROR before each of "
                  f"{n_anchors} anchors (R/CX/M)")
            noisy = inject_per_op_transport_dephasing(noisy, results, noise_params)
        else:
            print(f"  Transport mode: NONE -- no transport-time dephasing inserted")

        # Count instruction types after noise injection
        op_counts = {}
        for inst in noisy:
            op_counts[inst.name] = op_counts.get(inst.name, 0) + 1
        print(f"  Noisy circuit: num_qubits={noisy.num_qubits}, instructions={sum(op_counts.values())}")
        print(f"  Op breakdown (noisy):")
        for name in sorted(op_counts):
            print(f"    {name}: {op_counts[name]}")

        # Detector sampling under noise
        print(f"\n  Detector sampling under noise ({args.noise_shots} shots)...")
        try:
            sampler = noisy.compile_detector_sampler()
            det_samples, obs_samples = sampler.sample(
                args.noise_shots, separate_observables=True)
            n_det = det_samples.shape[1]
            det_fire_rate = det_samples.mean(axis=0)  # per-detector fire rate
            obs_flip_rate = obs_samples.mean(axis=0)
            print(f"    det shape: {det_samples.shape}")
            print(f"    detector fire rate: min={det_fire_rate.min():.4f}, "
                  f"max={det_fire_rate.max():.4f}, mean={det_fire_rate.mean():.4f}")
            print(f"    observable flip rate: {obs_flip_rate}")
        except Exception as e:
            print(f"    [FAIL] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1

        if args.dump_noisy:
            out_path = Path(args.dump_noisy)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(str(noisy))
            print(f"\n  Saved noisy: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
