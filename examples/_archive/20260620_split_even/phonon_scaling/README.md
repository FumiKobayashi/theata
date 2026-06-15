# Phonon scaling vs code distance

Maximum transversal (and axial) phonon count at MS gate execution time,
plotted as a function of code distance d for the snake_reuse1 family
(single LIZ compiler).

## Data (from `examples/single_liz/dX_reuse1/result.json`)

| d | n_data | max_transversal | max_axial | par_time_ms |
|---|--------|-----------------|-----------|-------------|
| 3 | 9 | 4.030 | 594.89 | 30.01 |
| 5 | 25 | 21.403 | 3692.89 | 161.80 |
| 7 | 49 | 69.286 | 12839.37 | 540.68 |
| 9 | 81 | 199.436 | 38068.64 | 1406.25 |

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
