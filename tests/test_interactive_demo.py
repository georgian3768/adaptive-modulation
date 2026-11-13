# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Unit tests for interactive_demo.py's simulation core (headless layer).

Run from the project root with:
    .venv/bin/python -m unittest discover -s tests
"""

import unittest

import numpy as np

from interactive_demo import FrameConfig, LinkSim, simulate_frame


def _cfg(**kwargs):
    """FrameConfig with wobble off (deterministic set-point) by default."""
    base = dict(distance=1.0, wobble=False, frame_bits=4096)
    base.update(kwargs)
    return FrameConfig(**base)


class TestLinkHighSnr(unittest.TestCase):
    def test_reaches_64qam_and_link_ok(self):
        sim = LinkSim(seed=1)
        res = None
        for _ in range(3):
            res = sim.step(_cfg(distance=1.0))
        # Frame 0 boots at bpsk (latency), frames 1-2 run the upgraded mode.
        self.assertEqual(res.mode_tx, "64qam")
        self.assertLess(res.ber_measured, 1e-3)
        self.assertTrue(res.link_ok)
        self.assertEqual(res.throughput_bps, 6e6)


class TestLinkLowSnr(unittest.TestCase):
    def test_stays_bpsk_and_link_out(self):
        sim = LinkSim(seed=1)
        res = None
        for _ in range(3):
            res = sim.step(_cfg(distance=15.0))
        self.assertEqual(res.mode_tx, "bpsk")
        self.assertFalse(res.link_ok)
        self.assertGreaterEqual(res.ber_measured, 1e-3)
        self.assertEqual(res.throughput_bps, 0.0)


class TestEstimatorAccuracy(unittest.TestCase):
    def test_estimate_within_2db_of_truth_awgn(self):
        # Mid-SNR AWGN frames: the pilot estimator's error stays inside
        # +-2 dB (the standalone check measures ~0.3 dB RMS).
        rng = np.random.default_rng(5)
        cfg = _cfg(distance=4.0)
        errs = []
        for i in range(5):
            res = simulate_frame(cfg, "qpsk", rng, frame=i)
            errs.append(abs(res.est_err_db))
        self.assertLessEqual(max(errs), 2.0)


class TestOneDwellLatency(unittest.TestCase):
    def test_frame1_transmits_fallback_then_upgrades(self):
        sim = LinkSim(seed=1)
        res = sim.step(_cfg(distance=1.0))
        # The frame is on the air before any estimate exists: robust boot.
        self.assertEqual(res.mode_tx, "bpsk")
        # ...but its own pilot estimate already selected 64-QAM for frame 2.
        self.assertEqual(res.mode_next, "64qam")
        self.assertEqual(sim.selector.mode, "64qam")


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_results(self):
        cfg = _cfg(distance=6.0, wobble=True, fading=True)
        sim_a, sim_b = LinkSim(seed=9), LinkSim(seed=9)
        for _ in range(3):
            ra = sim_a.step(cfg)
            rb = sim_b.step(cfg)
            self.assertEqual(ra.mode_tx, rb.mode_tx)
            self.assertEqual(ra.es_true_db, rb.es_true_db)
            self.assertEqual(ra.es_est_db, rb.es_est_db)
            self.assertEqual(ra.n_err, rb.n_err)
            self.assertEqual(ra.ber_measured, rb.ber_measured)
            np.testing.assert_array_equal(ra.rx_syms, rb.rx_syms)


if __name__ == "__main__":
    unittest.main()
