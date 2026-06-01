# Distance scan: d=3 / d=5 / d=7

Logical error rate p_L as a function of code distance d for the snake_reuse1
family (single-LIZ compiler, 3 rounds of QEC, MWPM decoding with pymatching).

## Files
- `plot.png` — p_L vs d with 1-sigma error bars (log scale on y)
- `data.json` — raw numbers (errors, shots, p_L, stderr)

## Key finding: V-shape (non-monotonic)

| d | p_L (3 rounds) |
|---|----------------|
| 3 | ~1.4e-3        |
| 5 | ~6e-4          |
| 7 | ~1.3e-3        |

- d=3 → d=5: improvement (~2.3x), consistent with surface code's distance-dependent protection
- d=5 → d=7: degradation (statistically significant). Transport-time dephasing
  scales with qubit count and stabilizer count, eventually overwhelming the
  exponential suppression the code is supposed to provide.

## Interpretation

Within the *current* single-LIZ snake compiler, larger d brings:
- More qubits to shuttle (transport time grows)
- More stabilizers to measure (wall-clock per round grows)

The resulting transport-time idle dephasing (~17-20 ms per round for d=3, more for
larger d) is the dominant error source. Surface code's exponential protection
requires the dominant error sources to scale slowly with d — which is not the
case here.

This is the strongest empirical justification for the multi-LIZ direction:
parallelizing transport across LIZs should restore d-scaling of p_L.

## Caveats
- p_MS uses a placeholder rate (phonon-based formula not calibrated against
  real device); absolute p_L values are not reliable, only relative trends.
- All other gate error rates are from the preliminary specification (5/26
  meeting): p_R=1e-4, p_meas=3.2e-3, p_reset=8e-4, Γ_φ=3.0 /s.
