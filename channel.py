# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Channel model: path loss, flat Rayleigh fading, AWGN.

Implemented so far:
    * path loss:        log-distance model -> linear amplitude gain
    * flat fading:      frequency-flat (block-constant) Rayleigh multiplier
    * AWGN:             the noise block (add_awgn / add_awgn_samples)

SNR bookkeeping used everywhere in this project
-----------------------------------------------
The user-facing quantity is Eb/N0 (bit energy / noise power spectral
density), the standard x-axis for BER plots. The BER formulas are written in
terms of Es/N0 (symbol energy / N0). The two are related by the number of
bits per symbol k:

    Es/N0 = k * (Eb/N0)          (linear),   k = bits per symbol

For BPSK, k = 1 so Es/N0 = Eb/N0. Higher-order modulations (QPSK k=2, 16-QAM
k=4, 64-QAM k=6) get the same Eb/N0 only when Es is k times larger.

Noise power normalization - the part that is easy to get wrong
--------------------------------------------------------------
Complex AWGN n(t) has power spectral density N0/2 in EACH of its two real
dimensions (in-phase and quadrature), so the total complex noise power is
N0/2 + N0/2 = N0. When we sample a symbol we therefore generate
Re(n), Im(n) ~ N(0, N0/2) independently.

Given the unit-energy convention (Es = E[|s|^2] = 1) and a desired Eb/N0:

    eb_n0_lin = 10**(eb_n0_db / 10)
    es_n0_lin = eb_n0_lin * k
    N0        = Es / es_n0_lin          # N0 carries units of "energy per symbol"
    noise     = sqrt(N0/2) * (randn + 1j*randn)   # variance N0/2 per dimension

If you forget the /2 (variance N0 instead of N0/2) every plotted BER curve
shifts ~3 dB off from theory; if Es != 1 and you don't normalize, the shift
is arbitrary. Both are classic silent bugs.
"""

import numpy as np


def path_loss(distance, d_ref=1.0, exponent=2.0, ref_loss_db=0.0):
    """Log-distance path-loss model, returned as a linear amplitude gain.

        L(d) [dB] = ref_loss_db + 10 * exponent * log10(d / d_ref)

    with reference distance d_ref (valid for d >= d_ref), reference loss
    ref_loss_db at d_ref, and path-loss exponent ~2 free space, ~2.7-4 for
    obstructed/urban links. d is clamped to d_ref so the model never returns
    a gain > 1.

    Returns a = 10^(-L/20): the factor that multiplies a complex baseband
    signal. Power goes as |a|^2 (a is an amplitude/voltage gain).
    """
    d = max(float(distance), float(d_ref))
    loss_db = float(ref_loss_db) + 10.0 * float(exponent) * np.log10(d / float(d_ref))
    return float(10.0 ** (-loss_db / 20.0))


def noise_power_from_eb_n0(eb_n0_db, bits_per_symbol, symbol_energy=1.0):
    """Return the total complex noise power N0 for a requested Eb/N0 in dB.

    N0 = Es / (k * (Eb/N0)_linear) with k = bits per symbol. For complex AWGN
    the per-dimension variance is N0/2 (in-phase and quadrature each get half).
    The symbol energy is passed in explicitly so sample-rate callers can
    reference it to the original symbol stream (not to the upsampled
    waveform, whose mean power is Es/sps).
    """
    eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
    es_n0_lin = eb_n0_lin * bits_per_symbol
    return symbol_energy / es_n0_lin


def add_awgn(symbols, eb_n0_db, bits_per_symbol=1, rng=None, symbol_energy=None):
    """Add complex Gaussian noise achieving the requested Eb/N0 in dB.

    symbols:          complex symbol array (any modulation). Energy is
                      measured from the data itself (Es = mean(|s|^2)), so
                      the caller does not need unit-energy symbols - but the
                      Eb/N0 interpretation is only meaningful for the symbol
                      energies the transmitter actually sends.
    eb_n0_db:         target Eb/N0 in decibels.
    bits_per_symbol:  k, number of bits each symbol carries (BPSK=1,
                      QPSK=2, 16-QAM=4, 64-QAM=6). Converts Eb/N0 -> Es/N0.
    rng:              optional np.random.Generator for reproducible runs.
    symbol_energy:    optional Es override. Pass the UNFADED symbol energy
                      when `symbols` have already been faded: measuring it
                      from the faded data would give |g|^2 * Es and silently
                      cancel the fading in the noise level. With fading,
                      Eb/N0 therefore keeps its average-Received meaning.
    out:              received symbols r = s + n, same shape as input.

    Returns the noisy symbols and the noise power N0 actually applied.
    """
    symbols = np.asarray(symbols)
    if symbols.size == 0:
        return symbols.copy()

    if rng is None:
        rng = np.random.default_rng()

    # --- convert dB to linear ratios ---------------------------------------
    eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
    es_n0_lin = eb_n0_lin * bits_per_symbol

    # --- measure actual symbol energy instead of assuming Es = 1 ------------
    # This makes the function robust to any future scaling and is the honest
    # definition: Es/N0 uses the energy the transmitter really launches.
    # (see `symbol_energy` override above for the faded-signal corner case)
    if symbol_energy is None:
        es = float(np.mean(np.abs(symbols) ** 2))
    else:
        es = float(symbol_energy)

    # --- noise power spectral density and per-dimension variance -------------
    n0 = es / es_n0_lin          # total complex noise power per symbol
    noise_var_per_dim = n0 / 2.0 # split equally between I and Q branches
    sigma = np.sqrt(noise_var_per_dim)

    # --- generate complex noise ----------------------------------------------
    noise = sigma * (rng.standard_normal(symbols.shape)
                     + 1j * rng.standard_normal(symbols.shape))

    return symbols + noise, n0


def add_awgn_samples(samples, n0, rng=None):
    """Add complex Gaussian noise of total power N0 to a sample-rate signal.

    samples:  the shaped waveform (samples at `sps` per symbol).
    n0:       total complex noise power, from noise_power_from_eb_n0() for the
              ORIGINAL symbol energy (Es), NOT from the upsampled waveform.

    Why the same N0 works at the sample rate: the receiver's matched filter
    is unit-norm (sum|h|^2 = 1), and matched filtering a white process by a
    unit-norm filter leaves its variance unchanged. So noise with N0/2 per
    dimension per sample appears with the SAME N0/2 per dimension at the
    decision point, independent of samples-per-symbol. This is what keeps the
    pulse-shaped chain bit-exact against the symbol-level chain; the
    pulse-shaping regression check in main.py verifies it numerically.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(n0 / 2.0)   # N0/2 per dimension, per sample
    noise = sigma * (rng.standard_normal(samples.shape)
                     + 1j * rng.standard_normal(samples.shape))
    return samples + noise


def apply_flat_fading(signal, rng=None, seed=None):
    """Rayleigh flat-fading amplitude multiplier: signal * alpha.

    alpha = sqrt((x^2 + y^2)/2) with x, y ~ N(0,1): a real, positive
    Rayleigh-distributed factor drawn INDEPENDENTLY for every array element.
    E[alpha^2] = 1, so the average received energy equals the transmitted
    energy and the x-axis Eb/N0 of a BER sweep keeps its usual meaning as
    AVERAGE Eb/N0 (noise is always set from the unfaded energy - see the
    symbol_energy override in add_awgn).

    Returns (faded, alpha) so the caller can undo the gain at the receiver:
    coherent detection with ideal AGC divides the received symbols by alpha
    before demodulation. That is what the classic Rayleigh-fading BER
    integrals assume (P_e is computed at the instantaneous SNR
    gamma = Eb/N0 * alpha^2) - and it matters: the decision lattice is fixed
    at the unit-energy constellation, so without the normalization the
    demapper would start folding attenuated outer levels toward the origin
    (a no-noise BER floor). Thresholds at zero (BPSK) are amplitude-robust,
    which is why the alpha normalization is invisible there.

    Why per-symbol draws instead of one gain per block: frequency-flat
    fading whose coherence time covers a whole Monte-Carlo batch gives each
    batch a single alpha, so the min_errors protocol would sample a handful
    of alpha draws and never converge. Averaging the error rate over many
    independent per-symbol fades reproduces the SAME integral
    E_alpha[P_e(Eb/N0 * alpha^2)] as slow block fading, but converges within
    one batch.

    alpha is real and positive (COHERENT reception): a real receiver also
    estimates and removes the channel PHASE (pilots/equalizer). A random
    complex gain of unknown phase would rotate the constellation off the
    decision grid - half the BPSK symbols would land inverted and BER would
    sit near 0.5 - which is a phase-estimation failure, not the fading
    penalty a coherent link sees.

    Apply this BEFORE the noise: fading multiplies the signal, then AWGN is
    added at the unfaded power level.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    # |N(0,1) + j*N(0,1)|/sqrt(2) is Rayleigh with E[alpha^2] = 1.
    alpha = np.sqrt((rng.standard_normal(signal.shape) ** 2
                     + rng.standard_normal(signal.shape) ** 2) / 2.0)
    return signal * alpha, alpha