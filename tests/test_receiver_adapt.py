# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Unit tests for receiver.py (pilot SNR estimator) and adapt.py (selector).

Run from the project root with:
    .venv/bin/python -m unittest tests.test_receiver_adapt -v
"""

import unittest

import numpy as np

from modem import modulate, BITS_PER_SYMBOL
from channel import add_awgn
from receiver import estimate_snr, SNR_DB_CEIL
from adapt import select_modulation


class TestSelectModulation(unittest.TestCase):
    """The decision function is pure logic; feed it a fixed table."""

    def setUp(self):
        # Es/N0 thresholds illustrative of the project's grid; keys ascending
        # from the most robust scheme to the highest data rate.
        self.table = {"bpsk": 6.8,
                      "qpsk": 9.8,
                      "16qam": 16.5,
                      "64qam": 22.9}

    def test_below_all_falls_back(self):
        self.assertEqual(select_modulation(-5.0, self.table), "bpsk")

    def test_picks_highest_met(self):
        self.assertEqual(select_modulation(12.0, self.table), "qpsk")
        self.assertEqual(select_modulation(20.0, self.table), "16qam")
        self.assertEqual(select_modulation(30.0, self.table), "64qam")

    def test_exact_threshold_counts(self):
        self.assertEqual(select_modulation(9.84, self.table), "qpsk")

    def test_empty_table_rejected(self):
        with self.assertRaises(ValueError):
            select_modulation(10.0, {})


class TestSnrEstimator(unittest.TestCase):
    """Data-aided estimator accuracy against injected AWGN noise."""

    def _run(self, es_n0_db, n_pilots, scheme="qpsk", seed=0):
        k = BITS_PER_SYMBOL[scheme]
        rng = np.random.default_rng(seed)
        pilots = modulate(rng.integers(0, 2, n_pilots * k), scheme)
        rx, _ = add_awgn(pilots, es_n0_db - 10.0 * np.log10(k),
                         bits_per_symbol=k, rng=rng)
        return estimate_snr(rx, pilots)

    def test_accuracy_good_snr(self):
        # A long pilot block should nail a clear 12 dB Es/N0 channel.
        est = self._run(12.0, 20_000)
        self.assertAlmostEqual(est, 12.0, delta=0.3)

    def test_accuracy_low_snr(self):
        est = self._run(2.0, 20_000)
        self.assertAlmostEqual(est, 2.0, delta=0.5)

    def test_scale_robustness(self):
        # A constant channel amplitude must not change the Es/N0 estimate:
        # estimating Es/N0 12 dB with the received block scaled by 0.5.
        k = 2
        rng = np.random.default_rng(1)
        pilots = modulate(rng.integers(0, 2, 20_000 * k), "qpsk")
        rx, _ = add_awgn(pilots, 12.0 - 10.0 * np.log10(k),
                         bits_per_symbol=k, rng=rng)
        est_a = estimate_snr(rx, pilots)
        est_b = estimate_snr(rx * 0.5, pilots)
        self.assertAlmostEqual(est_a, est_b, delta=0.2)

    def test_clean_channel_clamped(self):
        # Noiseless runs clamp at the ceiling instead of returning Inf/NaN.
        k = 2
        pilots = modulate(np.zeros(64 * k, dtype=int), "qpsk")
        self.assertEqual(estimate_snr(pilots, pilots), SNR_DB_CEIL)

    def test_rejects_mismatched_blocks(self):
        with self.assertRaises(ValueError):
            estimate_snr(np.ones(8), np.ones(4))


if __name__ == "__main__":
    unittest.main()