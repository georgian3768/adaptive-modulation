# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Threshold-table gate: table print, BER corroboration, hysteresis no-chatter.

Verifies the adaptive-link decision chain on three independent checks:

1. Table print  -  the final threshold table carried by adapt.py (from the
                   symbol-level sweep) plus the up-/down-shift thresholds that
                   fence each scheme's dead band (default hysteresis gap 1 dB).

2. BER corroboration  -  for every threshold point, the scheme the switch
   moves INTO must already be safely under BER 1e-3 there. BER is re-measured
   with the symbol-level simulator (monte_carlo_ber, seed 42) AT the threshold
   SNR, so the check is not an interpolation guess:
        up-shift at thr(S)+0.5   -> enters S,        verify BER(S)
        down-shift at thr(S)-0.5 -> enters P<...,    verify BER(P)
   bpsk is the fallback and has no scheme below it to switch into.

3. Hysteresis no-chatter  -  drive the selector with an SNR trace that soaks
   +-0.5 dB around a crossing for 100 steps. The dead band is exactly +-0.5 dB,
   so the peaks touch the up-shift edge: the selector may upgrade ONCE when it
   first touches that edge, then must stick. Anything but 0 or 1 mode switches
   is a chattering failure.

Run:  .venv/bin/python verify_thresholds.py
"""

import sys

import numpy as np

from adapt import (THRESHOLDS_ES_N0_DB, AdaptiveModulation,
                   up_shift_thresholds, down_shift_thresholds)
from main import monte_carlo_ber
from modem import BITS_PER_SYMBOL

TARGET_BER = 1e-3
GAP = 1.0                       # hysteresis gap (dB) baked into adapt.py default
N_OSCILLATE_STEPS = 100

PASS = "PASS"
FAIL = "FAIL"


def ber_at_es(scheme, es_n0_db, seed=42):
    """Re-measure BER with the symbol-level simulator at an Es/N0 point."""
    eb_n0_db = es_n0_db - 10.0 * np.log10(BITS_PER_SYMBOL[scheme])
    return monte_carlo_ber(scheme, eb_n0_db, seed=seed)


def verify_thresholds():
    """Print the final (up/down) table and corroborate every threshold with the
    BER simulator at the exact SNR."""
    order = sorted(THRESHOLDS_ES_N0_DB, key=THRESHOLDS_ES_N0_DB.get)
    up = up_shift_thresholds(THRESHOLDS_ES_N0_DB, GAP)
    down = down_shift_thresholds(THRESHOLDS_ES_N0_DB, GAP)

    print("Final adaptive-modulation threshold table (Es/N0 dB, symbol-level sweep):")
    print(f"   {'scheme':>6}  {'base thr':>9}  {'up-shift':>9}  {'down-shift':>10}")
    for s in order:
        print(f"   {s:>6}  {THRESHOLDS_ES_N0_DB[s]:9.2f}  {up[s]:9.2f}"
              f"  {down[s]:10.2f}")

    print(f"\nBER corroboration at each threshold (target < {TARGET_BER:.0e}):")
    print(f"   {'threshold':>22}  {'switches into':>14}  {'SNR dB':>7}"
          f"  {'BER':>10}  verdict")
    ok = True
    prev = order[0]
    for s in order:
        k = BITS_PER_SYMBOL[s]

        # up-shift: enter s, so BER must be the scheme's own at up[s].
        ber = ber_at_es(s, up[s])
        good = ber < TARGET_BER
        ok &= good
        print(f"   {'up-shift -> ' + s:>22}  {s:>14}  {up[s]:7.2f}"
              f"  {ber:10.4g}  {PASS if good else FAIL}")

        # down-shift: leave s into the lower-rate scheme.
        if s == prev:
            # lowest scheme: bpsk is the fallback, nothing below it to enter.
            print(f"   {'down-shift <- ' + s:>22}  {'(fallback)':>14}  "
                  f"{down[s]:7.2f}  {'--':>10}  n/a")
        else:
            ber = ber_at_es(prev, down[s])
            good = ber < TARGET_BER
            ok &= good
            print(f"   {'down-shift <- ' + s:>22}  {prev:>14}  {down[s]:7.2f}"
                  f"  {ber:10.4g}  {PASS if good else FAIL}")
        prev = s
    return ok


def verify_no_chatter():
    """Oscillate +-0.5 dB around each crossing for 100 calls; count switches."""
    order = sorted(THRESHOLDS_ES_N0_DB, key=THRESHOLDS_ES_N0_DB.get)
    print(f"\nHysteresis no-chatter: oscillate +-{GAP / 2:.1f} dB around each"
          f" boundary for {N_OSCILLATE_STEPS} calls (dead band is +-{GAP / 2:.1f}"
          f" dB, so peaks touch the up-shift edge).")
    ok = True
    for lo, hi in zip(order, order[1:]):
        cross = THRESHOLDS_ES_N0_DB[hi]
        am = AdaptiveModulation(initial_mode=lo)
        switches = 0
        mode = am.mode
        for step in range(N_OSCILLATE_STEPS):
            snr = cross + (0.5 if step % 2 == 0 else -0.5)
            nxt = am.update(snr)
            if nxt != mode:
                switches += 1
                mode = nxt
        good = switches <= 1
        ok &= good
        print(f"   boundary {lo:>6} -> {hi:>6} (cross {cross:5.2f} dB): "
              f"{switches} switch(es), {PASS if good else FAIL}")
    return ok


def main():
    thr_ok = verify_thresholds()
    chat_ok = verify_no_chatter()
    verdict = PASS if (thr_ok and chat_ok) else FAIL
    print(f"\nOverall: {verdict}")
    return 0 if verdict == PASS else 1


if __name__ == "__main__":
    sys.exit(main())