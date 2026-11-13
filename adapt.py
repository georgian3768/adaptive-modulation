# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Modulation selection for the adaptive link.

Two layers:

1. select_modulation(snr, table)             - stateless: highest-order scheme
                                                whose threshold is met.
2. AdaptiveModulation (hysteresis selector)  - stateful: remembers the current
   mode and uses TWO thresholds per boundary - an up-shift (switch to the
   next higher scheme) and a down-shift (drop a scheme) sitting +-hysteresis/2
   dB around the BER-1e-3 crossing. The dead band between them stops the mode
   from chattering when the SNR hovers at a boundary.

The base thresholds are encoded from the symbol-level Monte-Carlo sweep,
reproduced by build_threshold_table.py: the minimum Es/N0 (dB) at which each
scheme's SIMULATED BER stays under 1e-3 for the rest of the sweep. The units
are Es/N0 because receiver.estimate_snr outputs Es/N0, so estimates and
thresholds live on the same scale (Es/N0 = Eb/N0 + 10*log10(k)). A real
deployment would add link-budget margin on top; the mechanism is what matters.

Threshold table semantics: {scheme: min_snr_db}, ascending from the most
robust scheme (lowest requirement) to the highest data rate.
"""

# Crossings measured from the symbol-level BER sweep (seed 42, 0..24 dB grid,
# BER < 1e-3 for the whole tail) - paste output of build_threshold_table.py.
import numpy as np

THRESHOLDS_ES_N0_DB = {
    "bpsk": 7.00,
    "qpsk": 10.01,
    "16qam": 17.02,
    "64qam": 22.78,
}


def select_modulation(snr_est_db, threshold_table):
    """Return the highest-data-rate modulation whose SNR requirement is met.

    threshold_table: dict {scheme: min_snr_db}, ascending from the most
    robust scheme (lowest requirement) to the highest data rate.
    snr_est_db:      channel SNR estimate in dB, same scale as the table.

    Stateless helper: fine for one-shot decisions, but a time-varying channel
    hovering at a boundary makes a stateless caller oscillate between two
    modes - that is exactly what AdaptiveModulation's hysteresis fixes.

    Returns the scheme with the largest min_snr_db <= snr_est_db; if even the
    lowest threshold is not met, returns that most robust scheme (fallback).
    """
    if not threshold_table:
        raise ValueError("select_modulation: threshold_table is empty")

    items = sorted(threshold_table.items(), key=lambda kv: kv[1])
    chosen = items[0][0]                      # fallback: most robust scheme
    for scheme, min_db in items:
        if snr_est_db >= min_db:
            chosen = scheme                   # keep upgrading while met
        else:
            break                             # sorted: nothing above helps
    return chosen


def link_availability(ber_per_dwell, target_ber=1e-3):
    """Fraction (percent) of the trace whose BER stays below ``target_ber``.

    Availability is the share of time the link is usable: any dwell whose bit
    error rate violates the target counts as down, exactly as in the
    throughput rule (throughput = 0 there). Handles a single scalar as a
    one-dwell trace (returns 100.0 or 0.0). The BER vector is a measurement
    input - usually the per-dwell output of the same Monte-Carlo chain that
    produced the threshold table.
    """
    ber = np.asarray(ber_per_dwell, dtype=float)
    if ber.size == 0:
        raise ValueError("link_availability: empty BER vector")
    return 100.0 * float(np.count_nonzero(ber < target_ber)) / ber.size


def up_shift_thresholds(threshold_table=THRESHOLDS_ES_N0_DB,
                        hysteresis_db=1.0):
    """{scheme: snr_db} at which the selector ALLOWS moving UP into that scheme."""
    return {s: thr + hysteresis_db / 2.0 for s, thr in threshold_table.items()}


def down_shift_thresholds(threshold_table=THRESHOLDS_ES_N0_DB,
                          hysteresis_db=1.0):
    """{scheme: snr_db} below which the selector MUST leave that scheme."""
    return {s: thr - hysteresis_db / 2.0 for s, thr in threshold_table.items()}


class AdaptiveModulation:
    """Modulation selector with hysteresis (no chattering near a boundary).

    State: remembers the last chosen mode. On each update():

        * upgrade greedily to the highest scheme whose up-shift(
          threshold + hysteresis_db/2) is met - a single high-SNR reading can
          take the link straight to 64-QAM;
        * fall back a single scheme only when the estimate drops below
          down_shift(current) = threshold - hysteresis_db/2;
        * otherwise keep the current mode.

    With the default 1 dB gap the two thresholds are 1 dB apart, so an SNR
    hovering inside [threshold - 0.5, threshold + 0.5] dB never flips the
    mode.
    """

    def __init__(self, threshold_table=None, hysteresis_db=1.0,
                 initial_mode=None):
        """threshold_table: {scheme: min_snr_db}; defaults to the encoded
        threshold table so a plain AdaptiveModulation() is immediately
        usable."""
        if threshold_table is None:
            threshold_table = THRESHOLDS_ES_N0_DB
        if not threshold_table:
            raise ValueError("threshold_table is empty")
        # descending data-rate, ascending requirement: (scheme, threshold)
        self._items = sorted(threshold_table.items(), key=lambda kv: kv[1])
        self._index = {scheme: i for i, (scheme, _) in enumerate(self._items)}
        self.hysteresis_db = float(hysteresis_db)
        self.mode = (initial_mode if initial_mode is not None
                     else self._items[0][0])   # scheme of the lowest threshold

    def update(self, snr_est_db):
        """Feed a new SNR estimate; returns (and stores) the next mode.

        Upgrade is greedy but final: climbs from the current mode straight to
        the highest-order scheme whose up-shift threshold is met, so the link
        can lock 64-QAM in one call when the SNR justifies it. Stepping back
        down is one level per call (the crossings are several dB apart; a
        momentary wobble must not dump the mode to the floor).
        """
        cur = self._index[self.mode]
        half = self.hysteresis_db / 2.0

        # Upgrade greedily: each candidate needs its threshold beaten by half
        # the hysteresis gap. No mode is skipped in the process.
        j = cur
        while j + 1 < len(self._items) and \
                snr_est_db >= self._items[j + 1][1] + half:
            j += 1
        if j != cur:
            self.mode = self._items[j][0]
            return self.mode

        # Otherwise drop one level when the CURRENT mode's own threshold fades
        # by half the hysteresis gap. The lowest mode has nothing below it.
        if cur > 0 and snr_est_db < self._items[cur][1] - half:
            self.mode = self._items[cur - 1][0]
        return self.mode