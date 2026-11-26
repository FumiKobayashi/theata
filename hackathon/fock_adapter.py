# -*- coding: utf-8 -*-
"""Adapter: teammate's Fock-resolved MS channel table -> THeatA noise injection.

Replaces the 0.25-grid PauliErrorLookup with exact-n_bar channels computed from
the Fock-resolved table  E[P] = sum_n P(n) E_n  (msgate_data/, validated against
Hirai's chi to ~2e-5).  The phonon-number distribution shape P(n) at fixed mean
becomes an explicit choice:

    shape="thermal"            Var = m + m^2   (what Hirai's table assumes)
    shape="poisson"            Var = m
    shape=("nb", k)            negative binomial, Var = m + m^2/k
                               (k=1 reproduces thermal, k<1 super-thermal)

`fock_rates(at_gates, FockChannelLookup(...))` returns the same tuple list as
exp_noise_injection.phonon_to_pauli_lookup_rates, so inject_noise_approach_b
consumes it unchanged.

Model caveat (teammate's README): E_n is outside the model's validity for
n >~ 100; distributions whose mass beyond the grid (n > 400) exceeds `tail_tol`
raise, instead of extrapolating.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent
MSGATE_SRC = HERE / "msgate_data" / "msgate" / "src"
for p in (MSGATE_SRC, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from msgate.fock import FockTable  # noqa: E402
from msgate.model import thermal_probs  # noqa: E402
from msgate.pauli import LABELS  # noqa: E402
from pauli_lookup import STIM_PAULI_ORDER  # noqa: E402

NPZ_DEFAULT = HERE / "msgate_data" / "msgate" / "results" / "fock_tau100us.npz"


class FockChannelLookup:
    """Exact-n_bar 15-Pauli-probability evaluation for a chosen P(n) shape."""

    def __init__(self, shape="thermal", npz_path=NPZ_DEFAULT, tail_tol=1e-6):
        self.tab = FockTable.load(npz_path)
        self.nmax = int(self.tab.ns.max())
        self.tail_tol = tail_tol
        if shape in ("thermal", "poisson"):
            self.shape, self.k = shape, None
        else:
            kind, k = shape
            assert kind == "nb" and k > 0
            self.shape, self.k = "nb", float(k)
        self._idx = [LABELS.index(p) for p in STIM_PAULI_ORDER]
        self._cache = {}

    @property
    def label(self):
        return self.shape if self.k is None else f"nb_k{self.k:g}"

    def _distribution(self, m):
        n = np.arange(self.nmax + 1)
        if self.shape == "thermal":
            p = thermal_probs(m, self.nmax + 1)
        elif self.shape == "poisson":
            from scipy.stats import poisson
            p = poisson.pmf(n, m)
        else:
            from scipy.stats import nbinom
            p = nbinom.pmf(n, self.k, self.k / (self.k + m))
        p = np.asarray(p, dtype=float)
        tail = 1.0 - p.sum()
        if tail > self.tail_tol:
            raise ValueError(
                f"{self.label}: P(n > {self.nmax}) = {tail:.2e} at mean {m:.3f} "
                f"exceeds tail_tol -- outside the Fock table's range")
        return p / p.sum()

    def probs15(self, n_bar):
        """[p_IX, ..., p_ZZ] (STIM_PAULI_ORDER) at exact mean n_bar."""
        key = round(float(n_bar), 6)
        if key not in self._cache:
            chi = self.tab.combine(self._distribution(max(key, 1e-6)))
            d = chi.diagonal().real
            d = np.clip(d, 0.0, None)  # spline can leave ~1e-9 negatives
            self._cache[key] = [float(d[i]) for i in self._idx]
        return self._cache[key]


def fock_rates(phonon_at_gates, lookup, phonon_scale=1.0):
    """Drop-in for phonon_to_pauli_lookup_rates: exact n_bar, no grid, no clip.

    Returns [(gate_name, probs15, n_bar_used, max_axial, max_transversal), ...]
    where n_bar_used is the exact (scaled) mean fed to the channel.
    """
    rates = []
    for g in phonon_at_gates:
        phonons = g["phonons"]
        max_ax = max(p[0] for p in phonons.values())
        max_tr = max(p[1] for p in phonons.values())
        n_input = max_tr * phonon_scale
        rates.append((g["gate"], lookup.probs15(n_input), n_input, max_ax, max_tr))
    return rates


if __name__ == "__main__":
    # sanity: thermal shape on grid points must reproduce the CSV lookup table
    from pauli_lookup import PauliErrorLookup
    csv_lookup = PauliErrorLookup()
    fock_th = FockChannelLookup("thermal")
    worst = 0.0
    for n in (0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0):
        a = np.array(csv_lookup.lookup(n, mode="strict"))
        b = np.array(fock_th.probs15(n))
        worst = max(worst, np.abs(a - b).max())
        print(f"n_bar {n:5g}: max |p_csv - p_fock| = {np.abs(a - b).max():.2e}")
    print(f"worst over grid points: {worst:.2e}")
