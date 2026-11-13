# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Receive-side signal processing: pilot-based (data-aided) SNR estimation.

Why data-aided (known pilots)?
------------------------------
The adaptive transmitter needs an Es/N0 estimate per control period to pick
the next frame's modulation, and the estimator is data-aided: the
transmitter prepends a short block of KNOWN unit-energy pilot symbols (QPSK
here) to the frame. Three numbers fall out of correlating with them:

    signal   Es_hat = |mean_k r_k . conj(p_k)|^2 / mean_k|p_k|^2
    total    Ptot   = mean_k |r_k|^2
    noise    N0_hat = Ptot - Es_hat           (unbiased for the noise)

with r_k = h p_k + n_k. Es_hat is the correlator estimate of the received
signal power (biased HIGH by only N0/N for N pilots) and N0_hat subtracts it
from the total received power, so it is an unbiased noise estimate. Their
ratio is Es/N0 and is independent of any constant channel amplitude h and of
the pilot constellation as long as mean|p|^2 = 1.

Data-aided beats blind estimation because it never has to first *decide* the
data: at precisely the low-SNR / deep-fade frames where decisions are
unreliable (and where a modulation switch matters most), a blind estimator
couples its own decision errors back into the SNR it reports. Known pilots
are the cheap, standard field solution and need no sophisticated statistical
methods - a deliberately simple maximum-likelihood-style block estimate that
is easy to verify against the injected noise (see snr_estimator_check.py).

Caveat: with a finite pilot block N0_hat can go slightly negative when the
frame is essentially noiseless; the result is floored in dB so the selector
in adapt.py never sees a nonsensical value.
"""

import numpy as np

# The estimator cannot resolve SNRs outside these bounds; the clamp also
# prevents N0_hat ~ 0 from blowing up the ratio.
SNR_DB_FLOOR = -10.0
SNR_DB_CEIL = 40.0


def estimate_snr(rx_pilots, tx_pilots):
    """Estimate the received Es/N0 (dB) from a known pilot block.

    rx_pilots:  received complex pilot symbols (after AGC/equalization, before
                demodulation - a constant channel amplitude scale is fine).
    tx_pilots:  the transmitted pilot symbols, known to the receiver; any
                constellation with mean|p_k|^2 = 1 (e.g. QPSK) works.

    Returns Es/N0 in dB, clamped to [SNR_DB_FLOOR, SNR_DB_CEIL].
    """
    r = np.asarray(rx_pilots)
    p = np.asarray(tx_pilots)
    if r.size == 0 or r.size != p.size:
        raise ValueError("pilot blocks must be non-empty and of equal length")

    # |h|^2 averaged over the block: the correlator output normalised by the
    # pilots' own power so the estimate is right at any |p|^2 scale.
    mean_p2 = float(np.mean(np.abs(p) ** 2))
    signal = abs(np.mean(r * np.conj(p))) ** 2 / mean_p2
    total = float(np.mean(np.abs(r) ** 2))
    noise = total - signal

    if signal <= 0.0 or noise <= 0.0:
        # No measured power in the block or a finite-N noise glitch: report
        # the extreme seen last, clamped. The selector only needs the band.
        if total <= 0.0:
            return SNR_DB_FLOOR
        return SNR_DB_CEIL

    return float(np.clip(10.0 * np.log10(signal / noise),
                         SNR_DB_FLOOR, SNR_DB_CEIL))