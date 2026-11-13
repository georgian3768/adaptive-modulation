# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Unit tests for the adaptive selector (adapt.py): base thresholds, the
hysteresis state machine, and the no-chatter guarantee at a boundary.

Run from the project root with:
    .venv/bin/python -m unittest tests.test_adapt -v
"""

import unittest

import numpy as np

from adapt import (THRESHOLDS_ES_N0_DB, select_modulation,
                   AdaptiveModulation, up_shift_thresholds,
                   down_shift_thresholds, link_availability)


class TestSelectModulation(unittest.TestCase):

    def test_upgrades_to_highest_met(self):
        self.assertEqual(select_modulation(8.0, THRESHOLDS_ES_N0_DB), "bpsk")
        self.assertEqual(select_modulation(12.0, THRESHOLDS_ES_N0_DB), "qpsk")
        self.assertEqual(select_modulation(20.0, THRESHOLDS_ES_N0_DB), "16qam")
        self.assertEqual(select_modulation(25.0, THRESHOLDS_ES_N0_DB), "64qam")

    def test_falls_back_below_everything(self):
        self.assertEqual(select_modulation(-5.0, THRESHOLDS_ES_N0_DB), "bpsk")

    def test_empty_table_rejected(self):
        with self.assertRaises(ValueError):
            select_modulation(10.0, {})


class TestHysteresisTables(unittest.TestCase):

    def test_up_and_down_sit_one_db_apart(self):
        for scheme, thr in THRESHOLDS_ES_N0_DB.items():
            gap = (up_shift_thresholds(hysteresis_db=1.0)[scheme]
                   - down_shift_thresholds(hysteresis_db=1.0)[scheme])
            self.assertAlmostEqual(gap, 1.0, delta=1e-9)

    def test_thresholds_bracketed(self):
        up = up_shift_thresholds()
        down = down_shift_thresholds()
        for scheme, thr in THRESHOLDS_ES_N0_DB.items():
            self.assertLess(down[scheme], thr)
            self.assertGreater(up[scheme], thr)


class TestAdaptiveModulationChattering(unittest.TestCase):
    """The 16-QAM boundary (17.02 dB Es/N0, no hysteresis): with the default
    1 dB gap you must reach 17.52 to upgrade and may keep 16-QAM down to
    16.52. Hovering around the crossing must not flip the mode."""

    CROSS = THRESHOLDS_ES_N0_DB["16qam"]

    def _hover(self, lo, hi, calls=1000, initial="qpsk"):
        am = AdaptiveModulation(initial_mode=initial)
        history = []
        for i in range(calls):
            snr = lo if i % 2 == 0 else hi
            history.append(am.update(snr))
        return history

    def test_stateless_selector_would_chatter(self):
        # Sanity check that the problem is real: hovering +-0.3 dB around the
        # crossing flips a stateless table pick on every other call.
        lo, hi = self.CROSS - 0.3, self.CROSS + 0.3
        picks = [select_modulation(lo if i % 2 == 0 else hi,
                                   THRESHOLDS_ES_N0_DB)
                 for i in range(100)]
        self.assertIn("qpsk", picks)
        self.assertIn("16qam", picks)

    def test_no_chatter_hovering_below_crossing(self):
        # Hover +-0.3 dB around the crossing, approached from below: the
        # upgrade needs CROSS + 0.5 dB, never reached -> stays on qpsk
        # for every single call.
        history = self._hover(self.CROSS - 0.3, self.CROSS + 0.3)
        self.assertTrue(all(m == "qpsk" for m in history))

    def test_no_chatter_hovering_after_upgrade(self):
        # Push cleanly past the up-shift threshold (CROSS + 0.5), then hover
        # inside the dead band [CROSS - 0.5, CROSS + 0.5). With hysteresis the
        # mode must stick to 16-QAM; a stateless picker would alternate.
        am = AdaptiveModulation(initial_mode="qpsk")
        am.update(self.CROSS + 0.8)            # -> 16-QAM for good
        self.assertEqual(am.mode, "16qam")
        history = [am.update(self.CROSS - 0.3 if i % 2 == 0 else self.CROSS + 0.3)
                   for i in range(1000)]
        self.assertTrue(all(m == "16qam" for m in history))

    def test_graceful_fall_on_steady_drift(self):
        # Uniform downward drift starting inside the dead band then crossing
        # the down-shift threshold: exactly one transition, never oscillating.
        am = AdaptiveModulation(initial_mode="16qam")
        history = [am.update(self.CROSS + 0.4 - 0.01 * d) for d in range(200)]
        transitions = sum(a != b for a, b in zip(history, history[1:]))
        self.assertEqual(transitions, 1)
        self.assertEqual(history[0], "16qam")
        self.assertEqual(history[-1], "qpsk")

    def test_rejects_empty_table(self):
        with self.assertRaises(ValueError):
            AdaptiveModulation(threshold_table={})

    def test_default_starts_at_most_robust(self):
        # No initial_mode and no table: the selector boots in the scheme whose
        # threshold is lowest (bpsk), not in the threshold value itself.
        am = AdaptiveModulation()
        self.assertEqual(am.mode, "bpsk")


class TestLinkAvailability(unittest.TestCase):

    def test_all_below_is_full_availability(self):
        self.assertEqual(link_availability([1e-4, 5e-4, 9e-4]), 100.0)

    def test_all_above_is_zero(self):
        self.assertEqual(link_availability([1.1e-3, 2e-3, 9e-3]), 0.0)

    def test_partial_is_percent(self):
        # 4 of 8 dwells usable -> 50 %
        ber = [0.02, 0.01, 4e-4, 6e-4, 0.03, 8e-4, 0.05, 2e-4]
        self.assertAlmostEqual(link_availability(ber), 50.0)

    def test_boundary_counts_as_down(self):
        # BER exactly at the target is NOT below it: link is down.
        self.assertEqual(link_availability([1e-3]), 0.0)

    def test_scalar_input(self):
        self.assertEqual(link_availability(5e-4), 100.0)
        self.assertEqual(link_availability(5e-2), 0.0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            link_availability(np.array([]))


if __name__ == "__main__":
    unittest.main()