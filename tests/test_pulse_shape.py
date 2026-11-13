# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Unit tests for pulse_shape.py: RRC filter properties, upsampling and the
zero-ISI / round-trip guarantees the BER-neutrality of pulse shaping depends
on.

Run from the project root with:
    .venv/bin/python -m unittest tests.test_pulse_shape -v
"""

import unittest

import numpy as np

from pulse_shape import rrc_filter, upsample, tx_pulse_shape, rx_match_filter
from modem import modulate, constellation_points


class TestRrcFilter(unittest.TestCase):
    def test_tap_count(self):
        self.assertEqual(len(rrc_filter(0.22, span=6, sps=8)), 6 * 8 + 1)

    def test_unit_energy(self):
        for beta in (0.0, 0.22, 0.5, 1.0):
            h = rrc_filter(beta, span=6, sps=8)
            self.assertAlmostEqual(float(np.sum(np.abs(h) ** 2)), 1.0)

    def test_even_symmetry(self):
        h = rrc_filter(0.22, span=6, sps=8)
        np.testing.assert_array_almost_equal(h, h[::-1])

    def test_rolloff_range(self):
        for bad in (-0.1, 1.2):
            with self.assertRaises(ValueError):
                rrc_filter(bad, span=6, sps=8)


class TestUpsample(unittest.TestCase):
    def test_structure(self):
        symbols = np.array([3 + 0j, -1 + 2j])
        out = upsample(symbols, sps=4)
        self.assertEqual(len(out), 8)
        np.testing.assert_array_equal(out[::4], symbols)
        self.assertEqual(np.count_nonzero(out[1::4] + out[2::4] + out[3::4]), 0)


class TestZeroIsi(unittest.TestCase):
    def test_rrc_conv_rrc_is_nearly_nyquist(self):
        # RRC * RRC = raised cosine must equal 1 at lag 0 and (ideally) 0 at
        # every other symbol centre. Finite filter span truncates the RRC
        # tails, leaving a small residual (<2% of the signal for the default
        # parameters) that is far below the noise at any BER of interest.
        for beta in (0.22, 0.35, 1.0):
            with self.subTest(beta=beta):
                h = rrc_filter(beta, span=10, sps=8)
                rc = np.convolve(h, h)
                centre = len(rc) // 2
                self.assertAlmostEqual(rc[centre], 1.0, places=6)
                for m in range(-3, 4):
                    if m == 0:
                        continue
                    self.assertLess(abs(rc[centre + m * 8]), 0.02)


class TestRoundTrip(unittest.TestCase):
    def test_noiseless_bit_round_trip_all_schemes(self):
        # Shaped + matched-filtered, no noise: the residual truncation ISI is
        # far smaller than the decision margin, so every bit must come back
        # unchanged (symbols differ by ~1e-3 details, bits are exact).
        from modem import demodulate
        h = rrc_filter(0.22, span=10, sps=8)
        trim = 5  # span // 2 symbols dropped per edge
        for scheme in ("bpsk", "qpsk", "16qam", "64qam"):
            with self.subTest(scheme=scheme):
                constel = constellation_points(scheme)
                n_rep = 64
                symbols = np.tile(constel, n_rep)
                n_sym = len(symbols)
                wave = tx_pulse_shape(symbols, h, 8)
                rx = rx_match_filter(wave, h, 8, n_sym)[trim:-trim]
                np.testing.assert_array_equal(
                    demodulate(rx, scheme), demodulate(symbols[trim:-trim], scheme))


if __name__ == "__main__":
    unittest.main()