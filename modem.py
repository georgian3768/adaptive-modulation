# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Bit <-> symbol mapping for BPSK / QPSK / 16-QAM / 64-QAM (Gray-coded).

Conventions (used everywhere in this project)
--------------------------------------------
    * Every symbol has average energy Es = E[|s|^2] = 1 (unit-energy mapping).
      This is why the raw integer grids below are divided by a scale factor.
    * Symbols are complex: I (real) = in-phase, Q (imag) = quadrature.
    * The k bits of one symbol are laid out as [I bits | Q bits], with the
      first half of the group driving the I axis and the second half the Q
      axis. Within each half, the FIRST bit is the most significant bit.
    * Mapping is Gray-coded per axis: neighbouring constellation points (one
      grid step away in I or Q) differ in exactly ONE bit, so a symbol error
      at reasonable SNR is almost always a one-bit error.

Why the scale factors (sqrt(2), sqrt(10), sqrt(42))?
----------------------------------------------------
A Gray M-QAM axis is a PAM grid with evenly spaced levels, e.g. for 16-QAM
the I axis uses {-3,-1,+1,+3}; the Q axis uses the same grid. The average
symbol energy of the raw grid is the mean of |I+Q|^2 = 2 * mean(levels^2).
For 16-QAM: 2 * ((9+1+1+9)/4) = 2 * 5 = 10, hence the /sqrt(10). For QPSK
(2-PAM/axis): 2 * 1 = 2 -> /sqrt(2). For 64-QAM (8-PAM/axis): 2 * 21 = 42
-> /sqrt(42). Dividing by these factors forces Es = 1 for EVERY scheme, which
is what makes the SNR bookkeeping in channel.py correct.
"""

import numpy as np

BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}


def _gray_from_index(n):
    """Convert a plain integer n to its Gray-coded value: g = n ^ (n >> 1)."""
    return n ^ (n >> 1)


def _build_axis(b):
    """Build the per-axis lookup tables for a 2^b - level PAM axis.

    Returns (level_by_gray_int, boundaries, scale), where:
        level_by_gray_int[g]  = constellation level for Gray code g
                                (indexed by the Gray value -> dense 0..2^b-1,
                                so it can be used as a numpy lookup array)
        boundaries            = decision thresholds between levels
        scale                 = 1.0 for the I-only BPSK axis, sqrt(2) * axis
                                rms for the 2-axis QAM grids (per-symbol Es)

    Grid levels are {-L, -L+2, ..., +L} with L = 2^b - 1. The Gray code 0
    (bits all zero) is assigned to the MOST positive level (+L) so BPSK keeps
    the earlier convention: bit 0 -> positive, bit 1 -> negative.
    """
    n = 2 ** b
    ascending = np.arange(-(n - 1), n, 2, dtype=float)  # e.g. [-3,-1,1,3]
    levels = ascending[::-1].copy()                     # gray code 0 -> +L
    # level_by_gray_int[g] = PAM level for Gray bit pattern g. Build by
    # assigning the idx-th level to the position of its Gray code, so the
    # array index IS the Gray value read as bits (dense 0..n-1) and can be
    # used as a numpy lookup array in modulate().
    level_by_gray_int = np.empty(n)
    for idx in range(n):
        level_by_gray_int[_gray_from_index(idx)] = float(levels[idx])
    # Decision boundaries sit halfway between adjacent levels.
    boundaries = np.arange(-(n - 2), n - 1, 2, dtype=float)
    axis_rms = float(np.sqrt(np.mean(ascending ** 2)))
    return level_by_gray_int, boundaries, axis_rms


# Pre-build the axis tables once. b = bits per axis: BPSK/QPSK use 2-PAM
# (b=1), 16-QAM uses 4-PAM (b=2), 64-QAM uses 8-PAM (b=3).
_AXES = {b: _build_axis(b) for b in (1, 2, 3)}

_CONFIG = {}
for _name, _k in BITS_PER_SYMBOL.items():
    _b = 1 if _k == 1 else _k // 2
    _lut, _bounds, _axis_rms = _AXES[_b]
    # BPSK uses only the I axis (Q = 0), so the per-symbol scale is the axis
    # rms alone (which equals 1). The two-axis QAM grids get sqrt(2) * rms.
    _scale = _axis_rms if _k == 1 else np.sqrt(2.0) * _axis_rms
    _CONFIG[_name] = {"k": _k, "b": _b, "n": 2 ** _b,
                      "lut": _lut, "boundaries": _bounds, "scale": float(_scale)}


def _bin_to_int(bits2d, b):
    """Convert (..., b) bit arrays (first column = MSB) to integers 0..2^b-1."""
    weights = 1 << np.arange(b - 1, -1, -1)
    return (bits2d * weights).sum(axis=-1)


def _int_to_bin(values, b):
    """Convert integer arrays (0..2^b-1) to (..., b) bit arrays, MSB first."""
    values = np.asarray(values)
    out = np.zeros(values.shape + (b,), dtype=int)
    for shift in range(b):
        out[..., shift] = (values >> (b - 1 - shift)) & 1
    return out


def _gray_levels(bits2d, conf):
    """Map (n, b) bit arrays to the corresponding PAM levels on one axis."""
    gray_ints = _bin_to_int(bits2d, conf["b"])
    return conf["lut"][gray_ints]


def _levels_to_bits(axis_values, conf):
    """Quantize axis values to the nearest grid level, then to Gray bits.

    Closest level = the cell of `boundaries` the value falls into (binary
    search), which returns the index into the ascending level list. The
    constellation stored the levels in reverse order, so the Gray index is
    n - 1 - that index, and the Gray value is gray = n ^ (n >> 1).
    """
    i = np.searchsorted(conf["boundaries"], axis_values)
    gray = _gray_from_index(conf["n"] - 1 - i)
    return _int_to_bin(gray, conf["b"])


def modulate(bits, scheme):
    """Map a flat bit array to complex symbols for the given scheme."""
    if scheme not in BITS_PER_SYMBOL:
        raise ValueError(f"unknown scheme {scheme!r}")
    conf = _CONFIG[scheme]
    k = conf["k"]

    bits = np.asarray(bits).reshape(-1)
    n_sym = bits.size // k
    if n_sym * k != bits.size:
        raise ValueError(
            f"{scheme} needs bits in multiples of {k}; got {bits.size}")

    bits2d = bits[:n_sym * k].reshape(n_sym, k)
    if k == 1:  # BPSK: only the I axis, Q = 0
        ibits = bits2d[:, 0:1]
        ivals = _gray_levels(ibits, conf)
        qvals = np.zeros(n_sym)
    else:       # QPSK / QAM: first half of the group -> I, second half -> Q
        h = k // 2
        ivals = _gray_levels(bits2d[:, :h], conf)
        qvals = _gray_levels(bits2d[:, h:], conf)

    symbols = (ivals + 1j * qvals) / conf["scale"]
    return symbols


def demodulate(symbols, scheme):
    """Map noisy complex symbols back to bits (maximum-likelihood hard decision)."""
    if scheme not in BITS_PER_SYMBOL:
        raise ValueError(f"unknown scheme {scheme!r}")
    conf = _CONFIG[scheme]
    k = conf["k"]

    symbols = np.asarray(symbols).reshape(-1)
    # Rescale to the integer grid on which the decision boundaries are defined.
    scaled = symbols * conf["scale"]
    n_sym = scaled.size

    if k == 1:
        return _levels_to_bits(np.real(scaled), conf).reshape(-1)

    h = k // 2
    ibits = _levels_to_bits(np.real(scaled), conf)
    qbits = _levels_to_bits(np.imag(scaled), conf)
    return np.concatenate([ibits, qbits], axis=1).reshape(-1)


def constellation_points(scheme):
    """Return the full 2^k - point unit-energy constellation, one symbol each."""
    k = BITS_PER_SYMBOL[scheme]
    patterns = np.arange(2 ** k)
    return modulate(_int_to_bin(patterns, k).reshape(-1), scheme)


# --- named wrappers (kept for readability and for the BPSK unit checks) -----

def modulate_bpsk(bits):
    """Alias for modulate(..., 'bpsk')."""
    return modulate(bits, "bpsk")


def modulate_qpsk(bits):
    """Alias for modulate(..., 'qpsk')."""
    return modulate(bits, "qpsk")


def modulate_16qam(bits):
    """Alias for modulate(..., '16qam')."""
    return modulate(bits, "16qam")


def modulate_64qam(bits):
    """Alias for modulate(..., '64qam')."""
    return modulate(bits, "64qam")


def demodulate_bpsk(symbols):
    """Alias for demodulate(..., 'bpsk')."""
    return demodulate(symbols, "bpsk")


def demodulate_qpsk(symbols):
    """Alias for demodulate(..., 'qpsk')."""
    return demodulate(symbols, "qpsk")


def demodulate_16qam(symbols):
    """Alias for demodulate(..., '16qam')."""
    return demodulate(symbols, "16qam")


def demodulate_64qam(symbols):
    """Alias for demodulate(..., '64qam')."""
    return demodulate(symbols, "64qam")