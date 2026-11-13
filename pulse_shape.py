# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Root-raised-cosine (RRC) pulse shaping for the TX/RX chain.

What a pulse shaper does
------------------------
Two samples turn one symbol into a waveform:
  1. UPSAMPLE: insert SPS-1 zeros between symbols (symbol rate -> sample rate).
  2. FILTER: convolve with the RRC pulse. Each symbol becomes a brief
     (rolloff-limited-bandwidth) pulse; overlapping pulses form the waveform.

At the receiver the same RRC is used as the MATCHED filter. Two properties
make this BER-neutral versus the symbol-level model (verified by the
pulse-shaping regression check):
  * The cascade "RRC * RRC" is the raised-cosine (RC) pulse. An RC pulse
    is ZERO-ISI: its value is 1 at its own symbol centre and 0 at every other
    symbol centre, so under ideal timing the sampled output is essentially the
    transmitted symbol plus the filtered noise (pulse_shape.tx + rx return the
    centre samples on purpose).
  * The filter is unit-norm (sum |h|^2 = 1). Matched filtering white noise
    with a unit-norm filter leaves the noise variance unchanged, so the
    decision-point noise is exactly N0/2 per dimension - the same N0/2 the
    symbol-level chain injected. Increasing samples-per-symbol does not
    change the BER.

One honest caveat: the filter is finite (span symbols, "windowed"), and
truncating the RRC tails leaves a small residual ISI at symbol centres. With
the default span=10 the total residual ISI power is ~1e-5 of the signal
(about +0.2% extra noise at a BER ~1e-4 operating point), so the BER curves
stay unchanged within Monte Carlo precision. tests/test_pulse_shape.py pins
this number down. Increasing span reduces the residual (slowly, with the
classic 1/t^3 RRC tail decay).

Why unit-norm and not "peak 1"? If the pulse were not unit-norm the noise at
the decision point would NOT equal N0/2 - it would be N0/2 * sum|h|^2 - and
every curve would silently shift by 10*log10(sum|h|^2) dB.
"""

import numpy as np


def rrc_filter(rolloff, span, sps):
    """Return a unit-energy root-raised-cosine FIR tap vector.

    rolloff:  excess-bandwidth factor in [0, 1] (0.22 is a common radio value).
    span:     filter length in symbol periods, both sides (span * sps + 1 taps).
    sps:      samples per symbol (oversampling factor).

    The RRC impulse response (T = 1 symbol period, t measured in symbols):
        h(t) = [sin(pi(1-b)t) + 4b t cos(pi(1+b)t)] / [pi t (1 - (4bt)^2)]
    with b = rolloff, and the t->0 limit h(0) = 1 - b + 4b/pi. The taps are
    normalized afterwards so that sum(|h|^2) = 1 (unit energy).
    """
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError("rolloff must be in [0, 1]")
    if sps < 2:
        raise ValueError("sps (samples per symbol) must be >= 2")

    beta = float(rolloff)
    n_taps = int(span) * sps + 1
    # Sample times in symbol periods, symmetric around t = 0 (= centre tap).
    ts = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps
    h = np.zeros(n_taps)

    at_zero = np.isclose(ts, 0.0)
    h[at_zero] = 1.0 - beta + 4.0 * beta / np.pi

    # The denominator is 0 where 1 - (4bt)^2 = 0 -> t = +-1/(4b); the closed
    # form also has a limit there. (Only matters if a tap lands exactly on it.)
    at_critical = np.zeros(n_taps, dtype=bool)
    if beta > 0.0:
        at_critical = np.isclose(np.abs(ts), 1.0 / (4.0 * beta))
        h[at_critical] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))

    regular = ~(at_zero | at_critical)
    t = ts[regular]
    h[regular] = (
        np.sin(np.pi * (1.0 - beta) * t)
        + 4.0 * beta * t * np.cos(np.pi * (1.0 + beta) * t)
    ) / (np.pi * t * (1.0 - (4.0 * beta * t) ** 2))

    # Unit energy (see module docstring - this is the zero-ISI + bit-exact
    # noise-power guarantee).
    h /= np.sqrt(np.sum(np.abs(h) ** 2))
    return h


def upsample(symbols, sps):
    """Insert sps-1 zeros between symbols: symbol rate -> sample rate."""
    symbols = np.asarray(symbols)
    out = np.zeros(symbols.size * sps, dtype=complex)
    out[::sps] = symbols
    return out


def tx_pulse_shape(symbols, h_tx, sps):
    """Transmit: upsample the symbols and convolve with the RRC pulse."""
    return np.convolve(upsample(symbols, sps), h_tx)


def rx_match_filter(signal, h_rx, sps, n_symbols):
    """Receive: matched filter + downsample at ideal symbol centres.

    The RX filter is the TX RRC again (real and even -> time-reversal does
    nothing, so it is the matched filter). The t=0 tap is the centre tap of
    the array; each filter adds (len(h)-1)/2 taps of latency, so after BOTH
    filters a symbol n peaks at sample index n*sps + (len(h)-1). We therefore
    step through the filtered stream from that offset every sps samples.
    """
    y = np.convolve(signal, h_rx)
    onset = len(h_rx) - 1
    centres = onset + np.arange(n_symbols) * sps
    return y[centres]