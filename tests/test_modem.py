# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Unit tests for modem.py: hardcoded constellation points, round trips,
Es = 1 normalization, and Gray-coding adjacency.

Run from the project root with:
    .venv/bin/python -m unittest tests.test_modem -v
"""

import unittest

import numpy as np

from modem import (
    BITS_PER_SYMBOL,
    modulate, demodulate,
    modulate_bpsk, modulate_qpsk, modulate_16qam, modulate_64qam,
    demodulate_bpsk, demodulate_qpsk, demodulate_16qam, demodulate_64qam,
    constellation_points, _CONFIG, _int_to_bin,
)

_CONFIG_SCALE = {name: cfg["scale"] for name, cfg in _CONFIG.items()}


class TestBpsk(unittest.TestCase):
    def test_bit_mapping_signs(self):
        np.testing.assert_allclose(modulate_bpsk([0, 1, 0]), [+1, -1, +1])

    def test_round_trip(self):
        bits = np.array([1, 0, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(demodulate_bpsk(modulate_bpsk(bits)), bits)


class TestQpsk(unittest.TestCase):
    def test_constellation_points(self):
        # Layout per convention: bit0=I (0->+1, 1->-1), bit1=Q (0->+1, 1->-1).
        bits = [0, 0,  0, 1,  1, 0,  1, 1]
        d = 1 / np.sqrt(2)
        expected = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) * d
        np.testing.assert_allclose(modulate_qpsk(bits), expected, atol=1e-12)

    def test_round_trip(self):
        bits = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(demodulate_qpsk(modulate_qpsk(bits)), bits)


class Test16Qam(unittest.TestCase):
    def _grid(self):
        return constellation_points("16qam") * _CONFIG_SCALE["16qam"]

    def test_constellation_points(self):
        # I/Q axis Gray mapping (MSB first): 00->+3, 01->+1, 11->-1, 10->-3.
        bits = [0, 0, 0, 0,   0, 0, 1, 0,   0, 1, 0, 0,
                1, 0, 0, 0,   1, 0, 1, 1,   1, 1, 1, 1,   1, 1, 0, 1]
        d = 1 / np.sqrt(10)
        expected = np.array([
            3 + 3j,  3 - 3j,  1 + 3j,
            -3 + 3j, -3 - 1j, -1 - 1j, -1 + 1j,
        ]) * d
        np.testing.assert_allclose(modulate_16qam(bits), expected, atol=1e-12)

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 4096)
        np.testing.assert_array_equal(demodulate_16qam(modulate_16qam(bits)), bits)


class Test64Qam(unittest.TestCase):
    def test_constellation_points(self):
        # I/Q axis Gray mapping: 000->+7, 001->+5, 011->+3, 010->+1,
        #                        110->-1, 111->-3, 101->-5, 100->-7.
        bits = [0, 0, 0, 0, 0, 0,   0, 0, 0, 1, 0, 0,   0, 0, 1, 0, 0, 0,
                0, 1, 0, 0, 0, 0,   1, 0, 1, 0, 0, 0,   1, 0, 0, 0, 0, 0,
                1, 0, 1, 1, 0, 1,   1, 1, 0, 1, 1, 0]
        d = 1 / np.sqrt(42)
        expected = np.array([
            7 + 7j,  7 - 7j,  5 + 7j,  1 + 7j,
            -5 + 7j, -7 + 7j, -5 - 5j, -1 - 1j,
        ]) * d
        np.testing.assert_allclose(modulate_64qam(bits), expected, atol=1e-12)

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 4092)  # 4092 divisible by 6
        np.testing.assert_array_equal(demodulate_64qam(modulate_64qam(bits)), bits)


class TestNormalization(unittest.TestCase):
    def test_unit_energy_every_scheme(self):
        for scheme in BITS_PER_SYMBOL:
            constel = constellation_points(scheme)
            self.assertAlmostEqual(float(np.mean(np.abs(constel) ** 2)), 1.0)


class TestGrayCoding(unittest.TestCase):
    def test_adjacent_points_differ_in_one_bit(self):
        # For every scheme, any two symbols one grid step apart in I or Q
        # (the only errors a tolerance-level decision can make) must differ
        # in exactly one bit - that is the definition of Gray coding.
        for scheme, scale in _CONFIG_SCALE.items():
            constel = constellation_points(scheme) * scale  # integer grid coords
            k = BITS_PER_SYMBOL[scheme]
            m = 2 ** k
            bit_patterns = _int_to_bin(np.arange(m), k)
            with self.subTest(scheme=scheme):
                pairs_checked = 0
                for a in range(m):
                    for b in range(a + 1, m):
                        di = abs(constel[a].real - constel[b].real)
                        dq = abs(constel[a].imag - constel[b].imag)
                        is_grid_neighbour = (
                            (di == 2 and dq == 0) or (di == 0 and dq == 2))
                        if not is_grid_neighbour:
                            continue
                        pairs_checked += 1
                        hamming = int(np.count_nonzero(bit_patterns[a] != bit_patterns[b]))
                        self.assertEqual(hamming, 1,
                                         f"{scheme}: grid neighbours {a},{b} "
                                         f"differ in {hamming} bits")
                        self.assertEqual(hamming, 1)
                # Sanity: every point really does have grid neighbours.
                self.assertGreater(pairs_checked, 0)


if __name__ == "__main__":
    unittest.main()