# Distance scan: d=3 / d=5 (single-LIZ, PAULI_CHANNEL_2 lookup, strict)

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
