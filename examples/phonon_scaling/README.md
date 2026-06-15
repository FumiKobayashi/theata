# Phonon scaling vs code distance

Maximum transversal (and axial) phonon count at MS gate execution time,
plotted as a function of code distance d for the snake_reuse1 family
(single LIZ compiler).

## Data (from `examples/single_liz/dX_reuse1/result.json`)

| d | n_data | max_transversal | max_axial | par_time_ms |
|---|--------|-----------------|-----------|-------------|
| 3 | 9 | 2.995 | 470.73 | 26.92 |
| 5 | 25 | 14.944 | 2654.87 | 120.93 |
| 7 | 49 | 51.684 | 9701.62 | 376.53 |
| 9 | 81 | 133.740 | 25730.82 | 921.22 |

## Files
- `plot.png` — max transversal phonon vs d (linear scale, with lookup
  table max reference at n_bar = 20)
- `data.json` — full per-d numbers (axial, transversal, par/seq time)

## Notes
- The Pauli-twirled error probability lookup table
  (`src/pauli_twirled_error_probabilities.csv`) is sampled at n_bar ∈
  {0.01, 0.25, ..., 20.0}. For larger phonons (d ≥ 5 already has 1 MS
  exceeding 20, d ≥ 7 has many) the value is clipped to 20 with a
  runtime warning.
- Per-round wall time (`par_time_ms`) is included for reference but not
  plotted here; see `examples/distance_scan/` (noisy simulation) for the
  end-to-end logical error rate behavior.
- d=9 is included here even though it exceeds the lookup table — the
  phonon track itself is independent of the lookup table, so the data is
  trustworthy. Noise simulation for d=9 is intentionally not produced.
