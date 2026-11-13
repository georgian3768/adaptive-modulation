# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Derive the adaptive-modulation threshold table from simulated BER data.

Re-runs the symbol-level BER sweep (main.monte_carlo_ber, same
seed/min-errors protocol) over 0..24 dB and, for every scheme, finds the
minimum SNR at which the SIMULATED BER stays under the target (1e-3) for all
remaining points of the sweep. The "stays under for the rest of the sweep"
tail criterion is deliberately monotonic - it would not mis-fire on a single
stray noisy point.

Prints the resulting table in both Eb/N0 (the sweep's x-axis) and Es/N0
(= Eb/N0 + 10*log10(k)). The Es/N0 table is the one adapt.py carries on board
(THRESHOLDS_ES_N0_DB) because receiver.estimate_snr outputs Es/N0, so
estimates and thresholds live on the same scale.

Run:  .venv/bin/python build_threshold_table.py
"""

import numpy as np

from main import monte_carlo_ber, ALL_SCHEMES
from modem import BITS_PER_SYMBOL

TARGET_BER = 1e-3
SNR_GRID = np.arange(0.0, 25.0, 1.0)


def first_snr_stay_under(snr_db, ber, target=TARGET_BER):
    """Smallest snr_db[i] such that every ber[j] for j >= i is < target.

    The last above-target point is the one the tail must never repeat, so the
    crossing is the next sweep point after it. Returns None when the curve
    never drops below target (future schemes won't hit this).
    """
    below = np.asarray(ber) < target
    if not below.any():
        return None
    above = ~below
    last_above = int(np.flatnonzero(above)[-1]) if above.any() else -1
    return float(snr_db[last_above + 1])


def main():
    print(f"Re-running the symbol-level sweep "
          f"({SNR_GRID.size} points/scheme, always-used seed)...")
    print(f"Over TARGET_BER = {TARGET_BER:.0e}  ->  minimum SNR where the "
          f"simulated BER stays under it for every higher SNR:\n")

    print(f"{'scheme':>6}  {'k':>2}  {'Eb/N0 (dB)':>11}  {'Es/N0 (dB)':>11}")
    for scheme in ALL_SCHEMES:
        ber = np.array([monte_carlo_ber(scheme, e, seed=42) for e in SNR_GRID])
        eb_db = first_snr_stay_under(SNR_GRID, ber)
        k = BITS_PER_SYMBOL[scheme]
        es_db = None if eb_db is None else eb_db + 10.0 * np.log10(k)
        eb_s = f"{eb_db:7.1f}" if eb_db is not None else "   n/r"
        es_s = f"{es_db:7.1f}" if es_db is not None else "   n/r"
        print(f"{scheme:>6}  {k:>2}  {eb_s:>11}  {es_s:>11}")

    # python literal ready to paste into adapt.py
    print("\nCopy into adapt.py as THRESHOLDS_ES_N0_DB:")
    print("THRESHOLDS_ES_N0_DB = {")
    for scheme in ALL_SCHEMES:
        ber = np.array([monte_carlo_ber(scheme, e, seed=42) for e in SNR_GRID])
        eb_db = first_snr_stay_under(SNR_GRID, ber)
        k = BITS_PER_SYMBOL[scheme]
        es_db = eb_db + 10.0 * np.log10(k)
        print(f'    "{scheme}": {es_db:.2f},')
    print("}")


if __name__ == "__main__":
    main()