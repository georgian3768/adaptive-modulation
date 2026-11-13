# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

#!/usr/bin/env python3
"""demo.py - live adaptive-modulation demo over a receding-unit SNR trace.

Single-file and dependency-light (numpy + matplotlib only), so it runs
reliably during a live demo and does NOT import the project modules:

  * Es/N0 threshold table + the 1 dB hysteresis selector are inlined
    (identical logic to adapt.py);
  * per-dwell BER comes from the closed forms the sweep is validated against
    (plain math.erfc - no scipy), replacing the slow Monte-Carlo chain so the
    readout updates instantly; the report figures use the MC chain;
  * a live constellation panel for the currently selected mode (codebook +
    noisy received samples) repaints every dwell, flashing on mode switches;
  * a second panel draws the SNR trace as it is sailed, colour-coded by mode,
    with the four up-shift thresholds dotted.

Run:  .venv/bin/python demo.py [--dwells N] [--rt SECONDS] [--fast] [--save]
  --dwells N   dwell slots to simulate (default 40 = full trace)
  --rt SECONDS seconds between dwells for the live feel (default 0.35)
  --fast       no live animation / no sleep - print-only, for CI or demos
               without a display; still writes the final plot with --save
  --save       also write results/demo_constellation_final.png at the end

Ctrl-C ends the run early and prints the summary so far.
"""

import argparse
import math
import time

import numpy as np

HERE = __name__ == "__main__"

# ---- constants replicated from the full analyses ---------------------------
# Es/N0 (dB) where each scheme's SIMULATED BER stays below 1e-3
# (build_threshold_table.py). Selector semantics identical to adapt.py:
# greedily upgrade to the highest met up-shift (threshold + gap/2); step down
# one level only when the current threshold - gap/2 is missed.
TARGET_BER = 1e-3
HYSTERESIS_GAP_DB = 1.0
SCHEMES = ["bpsk", "qpsk", "16qam", "64qam"]
K = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}
THRESHOLDS_ES_N0 = {"bpsk": 7.00, "qpsk": 10.01, "16qam": 17.02, "64qam": 22.78}
SYMBOL_RATE = 1e6


class ModSelector:
    """Hysteresis-aware mode tracker (port of adapt.AdaptiveModulation)."""

    def __init__(self, table=THRESHOLDS_ES_N0, gap=HYSTERESIS_GAP_DB):
        self.items = sorted(table.items(), key=lambda kv: kv[1])
        self.index = {s: i for i, (s, _) in enumerate(self.items)}
        self.gap = float(gap)
        self.mode = self.items[0][0]

    def update(self, es_db):
        cur = self.index[self.mode]
        half = self.gap / 2.0
        j = cur
        while j + 1 < len(self.items) and \
                es_db >= self.items[j + 1][1] + half:
            j += 1
        if j != cur:
            self.mode = self.items[j][0]
            return self.mode
        if cur > 0 and es_db < self.items[cur][1] - half:
            self.mode = self.items[cur - 1][0]
        return self.mode


def receding_snr_trace(n_dwells=40, snr0=27.0, d_far=15.0, exponent=2.2,
                       wobble=0.4, seed=7):
    """Es/N0 (dB) per dwell, unit receding at constant radial velocity
    (same model as main.synthetic_snr_trace)."""
    rng = np.random.default_rng(seed)
    d = np.linspace(1.0, d_far, n_dwells)
    return snr0 - 10.0 * exponent * np.log10(d) + rng.normal(0.0, wobble, n_dwells)


def ber_theory(scheme, es_db):
    """Instant per-dwell BER (Es/N0 in dB) - the validated closed forms."""
    eb_db = es_db - 10.0 * math.log10(K[scheme])
    eb_lin = 10.0 ** (eb_db / 10.0)
    if scheme in ("bpsk", "qpsk"):
        return 0.5 * math.erfc(math.sqrt(eb_lin))
    m = 2 ** K[scheme]
    coef = (4.0 / K[scheme]) * (1.0 - 1.0 / math.sqrt(m))
    return coef * 0.5 * math.erfc(math.sqrt(3.0 * K[scheme] * eb_lin
                                            / (2.0 * (m - 1.0))))


def constellation(scheme):
    """Unit-energy (Es = 1) constellation points for one scheme."""
    if scheme == "bpsk":
        pts = [complex(-1, 0), complex(1, 0)]
    else:
        side = int(round(math.sqrt(2 ** K[scheme])))
        vals = list(range(-side + 1, side, 2))
        pts = [complex(x, y) for y in vals for x in vals]
    pts = np.asarray(pts, dtype=complex)
    pwr = np.mean(np.abs(pts) ** 2.0)          # normalize to Es = 1
    return pts / math.sqrt(pwr)


def noisy_samples(scheme, es_db, n=300, seed=None):
    """n received symbols at Es/N0 for the current dwell (AWGN only)."""
    pts = constellation(scheme)
    rng = np.random.default_rng(seed)
    es_lin = 10.0 ** (es_db / 10.0)
    sigma = math.sqrt(0.5 / es_lin)            # N0/2 per dimension, Es = 1
    idx = rng.integers(0, pts.size, n)
    return pts[idx] + sigma * (rng.standard_normal(n)
                               + 1j * rng.standard_normal(n))


class LivePlot:
    """Two live panels: constellation (selected mode) + SNR trace cursor.

    Falls back to draw-anyway mode when there is no interactive display, so
    --fast / CI still get a final figure via save().
    """

    def __init__(self, trace, interactive):
        self.interactive = interactive
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, (self.ax_const, self.ax_trace) = plt.subplots(
            1, 2, figsize=(12.5, 5.0))
        self.fig.suptitle("Adaptive-modulation live demo \u2014 unit receding "
                          "out of range", fontsize=12, fontweight="bold")

        # SNR trace panel with dotted up-shift thresholds
        n = trace.size
        self.ax_trace.plot(np.arange(n), trace, color="0.55", linewidth=1.0,
                           alpha=0.7)
        for scheme, thr in sorted(THRESHOLDS_ES_N0.items(),
                                  key=lambda kv: kv[1]):
            self.ax_trace.axhline(thr + HYSTERESIS_GAP_DB / 2.0,
                                  linestyle=":", color="0.75", linewidth=0.9)
            self.ax_trace.text(n * 1.005, thr + HYSTERESIS_GAP_DB / 2.0,
                               scheme, fontsize=7, color="0.4")
        self.ax_trace.set_xlabel("dwell")
        self.ax_trace.set_ylabel("Es/N0 (dB)")
        self.ax_trace.set_xlim(0, n)
        self.ax_trace.grid(True, alpha=0.3)
        self.cursor, = self.ax_trace.plot([], [], marker="o", ms=9,
                                          markerfacecolor="k",
                                          markeredgecolor="white")
        self.switch_text_trace = self.ax_trace.text(0.5, 0.92, "",
                                                    transform=self.ax_trace.transAxes,
                                                    ha="center", fontsize=9,
                                                    fontweight="bold",
                                                    color="#b30000")

        # constellation panel: fixed equal-aspect window wide enough for
        # 64-QAM's outer corners plus low-SNR noise tails
        self.ax_const.set_aspect("equal")
        self.ax_const.set_xlim(-3.0, 3.0)
        self.ax_const.set_ylim(-3.0, 3.0)
        self.ax_const.grid(True, alpha=0.3)
        self.ax_const.set_xlabel("I")
        self.ax_const.set_ylabel("Q")
        self.switch_text = self.ax_const.text(0.03, 0.95, "",
                                              transform=self.ax_const.transAxes,
                                              ha="left", va="top", fontsize=10,
                                              fontweight="bold", color="#b30000")
        self.codebook = None
        self.rx_scatter = None
        self.cb_scatter = None
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        if interactive:
            plt.ion()
            plt.show()

    def update(self, idx, es_db, mode, rx, switched_to=None):
        pts = constellation(mode)
        if self.cb_scatter is None:
            self.cb_scatter = self.ax_const.scatter(
                pts.real, pts.imag, s=70, color="0.35", zorder=3,
                label="codebook")
            self.rx_scatter = self.ax_const.scatter(
                rx.real, rx.imag, s=9, alpha=0.65, zorder=2, label="rx")
        else:
            self.cb_scatter.set_offsets(np.c_[pts.real, pts.imag])
            self.rx_scatter.set_offsets(np.c_[rx.real, rx.imag])
        self.ax_const.set_title(f"constellation: {mode}   "
                                f"Es/N0 = {es_db:5.2f} dB\n"
                                f"BER est = {ber_theory(mode, es_db):.2e}",
                                fontsize=10)
        self.cursor.set_data([idx], [es_db])
        self.switch_text.set_text(f"SWITCH \u2192 {switched_to}"
                                  if switched_to else "")
        self.switch_text_trace.set_text(f"dwell {idx + 1}  \u2192 {mode}"
                                        if switched_to else "")
        self.plt.draw()

    def pause(self, seconds):
        if self.interactive:
            self.plt.pause(seconds)

    def save(self, path):
        self.fig.savefig(path, dpi=150)


def fmt_mbps(value):
    return f"{value / 1e6:.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Live adaptive-modulation demo over a receding-unit SNR trace.")
    parser.add_argument("--dwells", type=int, default=40)
    parser.add_argument("--rt", type=float, default=0.35,
                        help="seconds between dwells (live feel)")
    parser.add_argument("--fast", action="store_true",
                        help="no animation/sleep, print-only")
    parser.add_argument("--save", action="store_true",
                        help="write results/demo_constellation_final.png")
    args = parser.parse_args(argv)
    if args.dwells < 2:
        parser.error("--dwells must be >= 2")

    import matplotlib
    headless = matplotlib.get_backend().lower() == "agg"

    trace = receding_snr_trace(args.dwells)
    # Interactive when a display exists; otherwise still build the figure when
    # --save was requested (Agg renders fine for a saved frame).
    plot = None
    if not headless or args.save:
        interactive = not (headless or args.fast)
        plot = LivePlot(trace, interactive)

    selector = ModSelector()

    rng = np.random.default_rng(2026)
    on_air = 0
    total_bits = 0
    print(f"adaptive live demo | {args.dwells} dwells | Rs = "
          f"{SYMBOL_RATE:.0e} symb/s | hysteresis gap {HYSTERESIS_GAP_DB:.1f} dB")
    print(f"dwell  Es/N0     mode    BER      link   thr[k]    running cum(Mbit)")

    try:
        for i, es_db in enumerate(trace):
            prev = selector.mode
            mode = selector.update(es_db)
            switched = mode if mode != prev else None
            ber = ber_theory(mode, es_db)
            ok = ber < TARGET_BER
            thr = K[mode] * SYMBOL_RATE if ok else 0.0
            if ok:
                on_air += 1
                total_bits += K[mode] * SYMBOL_RATE
            if plot is not None:
                rx = noisy_samples(mode, es_db, seed=int(rng.integers(0, 1 << 30)))
                plot.update(i, es_db, mode, rx, switched_to=switched)
                plot.pause(args.rt)
            elif not args.fast:
                time.sleep(args.rt)
            print(f"{i + 1:>4}   {es_db:5.2f}   {mode:>6}  {ber:8.1e}"
                  f"  {'OK  ' if ok else 'OUT '} {fmt_mbps(thr):>7}   "
                  f"{total_bits / 1e6:9.2f}")
    except KeyboardInterrupt:
        print("\n(stopped early)")

    availability = 100.0 * on_air / len(trace)
    print("\n--- summary ---")
    print(f"dwells on-air: {on_air}/{len(trace)} "
          f"({availability:.1f}%)")
    print(f"cumulative payload on-air: {total_bits / 1e6:.1f} Mbit")
    print(f"last mode: {selector.mode}")

    import os
    if args.save:
        os.makedirs("results", exist_ok=True)
        out = "results/demo_constellation_final.png"
        plot.save(out)
        print(f"final frame saved to {out}")
    elif plot is not None and plot.interactive:
        print("close the figure window to exit")
        plot.plt.ioff()
        plot.plt.show(block=True)


if __name__ == "__main__":
    main()