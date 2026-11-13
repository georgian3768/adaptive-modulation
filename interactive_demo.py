# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Interactive adaptive-modulation link demo (Tkinter GUI + simulation core).

Two layers live in one file:

1. Simulation core (no GUI/plot imports at module level, so tests can import
   this module headless): one FRAME is a complete pass through the real chain
   used in the report

       bits -> modulate -> [Rayleigh flat fade] -> RRC TX -> AWGN at the
       sample rate -> RRC matched filter -> downsample -> [ideal AGC] ->
       pilot-block SNR estimate -> demodulate -> count errors

   framed as [guard | 64 known QPSK pilots | data | guard]: the guards let
   the first/last TRIM symbols (edge transients of the pulse-shaping pair,
   same trim as monte_carlo_ber_shaped) be dropped without losing a pilot
   or data symbol.

   The selector (adapt.AdaptiveModulation) consumes the PILOT ESTIMATE of
   frame N to choose the modulation of frame N+1 - the one-dwell feedback
   latency of the block diagram in PRESENTATION.md. The estimator is the
   real data-aided one from receiver.py, so estimation error and chatter
   suppression by the hysteresis band are visible live, not idealised away.

2. GUI (tkinter + embedded matplotlib, imported only inside run_gui()):
   channel-story controls (distance -> log-distance path loss, AWGN/Rayleigh,
   set-point wobble, path-loss exponent, frame size), Step / Auto / Reset,
   TX-end and RX-end parameter + result panels, clean/noisy constellations,
   an Es/N0 history trace with up-shift thresholds, and a switch log.

Run:
    .venv/bin/python interactive_demo.py               # GUI (needs display)
    .venv/bin/python interactive_demo.py --headless 40  # print-only, CI-safe
    .venv/bin/python interactive_demo.py --selftest     # GUI smoke, ~2 s
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from adapt import (AdaptiveModulation, THRESHOLDS_ES_N0_DB,
                   link_availability)
from channel import (add_awgn_samples, apply_flat_fading,
                     noise_power_from_eb_n0)
from demo import HYSTERESIS_GAP_DB, SYMBOL_RATE, TARGET_BER, ber_theory
from modem import BITS_PER_SYMBOL, demodulate, modulate
from pulse_shape import rrc_filter, rx_match_filter, tx_pulse_shape
from receiver import estimate_snr

ALL_SCHEMES = ["bpsk", "qpsk", "16qam", "64qam"]
COLORS = {"bpsk": "#1f77b4", "qpsk": "#ff7f0e",
          "16qam": "#2ca02c", "64qam": "#d62728"}
# The orange qpsk banner reads better with dark text; white elsewhere.
BANNER_FG = {"qpsk": "#000000"}

RRC_ROLLOFF = 0.22
RRC_SPAN = 10
SPS = 8
TRIM = RRC_SPAN // 2
N_PILOTS = 64
SEED_DEFAULT = 42
AUTO_STEP = 0.35          # distance units per auto tick (40 ticks: 1 -> 15)
AUTO_MS = 150             # ~6 s for the full receding-unit ramp


@dataclass(frozen=True)
class FrameConfig:
    """Channel conditions + frame size for the next frame.

    distance:   units of the reference distance; set-point Es/N0 =
                snr0 - 10*exponent*log10(distance), i.e. the receding-unit
                story of synthetic_snr_trace (27 dB at 1 -> ~1 dB at 15).
    wobble:     +-0.4 dB per-frame jitter on the set-point, the same
                log-normal-ish wobble the offline trace uses.
    frame_bits: requested payload bits; the real count is truncated to a
                whole number of symbols (k varies with the mode).
    """

    distance: float = 1.0
    fading: bool = False
    wobble: bool = False
    frame_bits: int = 8192
    exponent: float = 2.2
    snr0: float = 27.0
    wobble_sigma: float = 0.4


@dataclass
class FrameResult:
    """Everything one frame proved - what the TX/RX panels display."""

    frame: int
    mode_tx: str                 # mode this frame was transmitted with
    mode_next: str               # chosen from this frame's pilot estimate
    k: int
    es_set_db: float             # set-point before wobble
    es_true_db: float            # set-point after wobble (noise is from this)
    es_est_db: float             # receiver's pilot-block estimate
    est_err_db: float            # est - true: estimator noise (+ fade)
    n0: float
    n_bits: int
    n_err: int
    ber_measured: float
    ber_closed: float            # AWGN closed form at es_true (reference)
    link_ok: bool                # measured BER below the 1e-3 target
    throughput_bps: float        # k*Rs while on air, else 0
    tx_syms: np.ndarray          # clean transmitted data symbols
    rx_syms: np.ndarray          # decision-point symbols (post-MF, post-AGC)
    sym_err_mask: np.ndarray     # per data symbol: any bit wrong


def set_point_es_n0(cfg):
    """Es/N0 (dB) the channel is currently set to, before wobble."""
    return cfg.snr0 - 10.0 * cfg.exponent * np.log10(max(cfg.distance, 1.0))


def simulate_frame(cfg, mode_tx, rng, frame=0):
    """Push one frame through the real chain; return every TX/RX quantity.

    The frame layout is [guard | pilots | data | guard] with TRIM guards, so
    after rx_match_filter(...)[TRIM:-TRIM] the receiver has exactly the
    known pilot block followed by the payload - no pilot or data symbol is
    ever taken from the filter edge-transient region.
    """
    k = BITS_PER_SYMBOL[mode_tx]
    es_set_db = set_point_es_n0(cfg)
    es_true_db = es_set_db
    if cfg.wobble:
        es_true_db += float(rng.normal(0.0, cfg.wobble_sigma))
    # N0 from the unfaded symbol energy (Es = 1): the average-Eb/N0
    # convention, exactly like monte_carlo_ber_shaped under fading.
    n0 = noise_power_from_eb_n0(es_true_db - 10.0 * np.log10(k), k,
                                symbol_energy=1.0)

    n_data = max(cfg.frame_bits // k, 1)
    n_bits = n_data * k
    data_bits = rng.integers(0, 2, n_bits)
    data_syms = modulate(data_bits, mode_tx)
    pilots = modulate(rng.integers(0, 2, 2 * N_PILOTS), "qpsk")
    guard = modulate(rng.integers(0, 2, 2 * TRIM), "qpsk")

    tx_syms = np.concatenate([guard, pilots, data_syms, guard])
    if cfg.fading:
        tx_syms, alpha = apply_flat_fading(tx_syms, rng=rng)
    else:
        alpha = None

    h = rrc_filter(RRC_ROLLOFF, RRC_SPAN, SPS)
    wave = tx_pulse_shape(tx_syms, h, SPS)
    wave = add_awgn_samples(wave, n0, rng=rng)
    rx = rx_match_filter(wave, h, SPS, tx_syms.size)[TRIM:-TRIM]
    if alpha is not None:
        # Ideal AGC over the kept region: first N_PILOTS entries are the
        # pilots, the rest the payload (guards already trimmed away).
        rx = rx / alpha[TRIM:-TRIM]

    rx_pilots, rx_data = rx[:N_PILOTS], rx[N_PILOTS:]
    es_est_db = estimate_snr(rx_pilots, pilots)

    rx_bits = demodulate(rx_data, mode_tx)
    err_bits = rx_bits != data_bits
    n_err = int(np.count_nonzero(err_bits))
    ber_measured = n_err / n_bits
    ber_closed = ber_theory(mode_tx, es_true_db)
    link_ok = ber_measured < TARGET_BER

    return FrameResult(
        frame=frame, mode_tx=mode_tx, mode_next=mode_tx, k=k,
        es_set_db=es_set_db, es_true_db=es_true_db,
        es_est_db=es_est_db, est_err_db=es_est_db - es_true_db,
        n0=n0, n_bits=n_bits, n_err=n_err,
        ber_measured=ber_measured, ber_closed=ber_closed,
        link_ok=link_ok,
        throughput_bps=(k * SYMBOL_RATE) if link_ok else 0.0,
        tx_syms=data_syms, rx_syms=rx_data,
        sym_err_mask=err_bits.reshape(n_data, k).any(axis=1),
    )


class LinkSim:
    """Stateful frame driver: selector + history + running link counters.

    step(cfg) transmits with the CURRENT selector mode, measures this
    frame's pilots, then feeds the estimate back - the next step() picks up
    the mode that estimate selected. Frame 0 therefore boots at the robust
    fallback (bpsk) even at 27 dB and upgrades one dwell later, which is the
    latency the block diagram promises.
    """

    def __init__(self, seed=SEED_DEFAULT):
        self.reset(seed)

    def reset(self, seed=SEED_DEFAULT):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.selector = AdaptiveModulation()
        self.frame = 0
        self.history = []
        self.switches = []       # (frame, prev, new, est_db)
        self.on_air = 0
        self.payload_bits = 0

    def step(self, cfg):
        prev = self.selector.mode
        res = simulate_frame(cfg, prev, self.rng, frame=self.frame)
        self.selector.update(res.es_est_db)
        res.mode_next = self.selector.mode
        if res.mode_next != prev:
            self.switches.append((res.frame, prev, res.mode_next,
                                  res.es_est_db))
        if res.link_ok:
            self.on_air += 1
            self.payload_bits += res.n_bits
        self.history.append(res)
        self.frame += 1
        return res

    @property
    def availability(self):
        if not self.history:
            return 0.0
        return link_availability([r.ber_measured for r in self.history])


def _fmt_ber(value):
    return f"{value:.1e}"


def _fmt_bits(n_bits):
    if n_bits >= 1_000_000:
        return f"{n_bits / 1e6:.2f} Mbit"
    if n_bits >= 1_000:
        return f"{n_bits / 1e3:.1f} kbit"
    return f"{n_bits} bit"


def _fmt_rate(bps):
    if bps <= 0.0:
        return "0 (link out)"
    return f"{bps / 1e6:.2f} Mbit/s"


def k_to_rate(k):
    return f"{k * SYMBOL_RATE / 1e6:.2f} Mbit/s"


def _results_text(res, sim, cfg):
    """The TX-end / RX-end panel content for one frame."""
    mode = res.mode_tx
    fading = "Rayleigh + ideal AGC" if cfg.fading else "AWGN"
    wobble = "on" if cfg.wobble else "off"
    switch = ""
    if res.mode_next != mode:
        switch = f"\n     next frame -> {res.mode_next} (from this estimate)"
    return (
        f"--- TX end --------------------------------\n"
        f"  {mode.upper()}  k={res.k}  Gray  Es=1\n"
        f"  frame #{res.frame}  {res.n_bits} bits "
        f"({res.n_bits // res.k} data syms)\n"
        f"  pilots {N_PILOTS} QPSK  guard {TRIM}+{TRIM}\n"
        f"  RRC beta={RRC_ROLLOFF}  span={RRC_SPAN}  sps={SPS}\n"
        f"  d={cfg.distance:.1f}  exp={cfg.exponent:.1f} -> "
        f"set {res.es_set_db:.1f} dB\n"
        f"  channel: {fading}  wobble {wobble}  N0={res.n0:.3f}\n"
        f"  rate if on air: {k_to_rate(res.k)}\n"
        f"--- RX end --------------------------------\n"
        f"  est Es/N0 {res.es_est_db:.1f} dB   true {res.es_true_db:.1f} dB"
        f"   err {res.est_err_db:+.1f} dB\n"
        f"  BER {_fmt_ber(res.ber_measured)}  "
        f"({res.n_err}/{res.n_bits} bits wrong)\n"
        f"  ref  {_fmt_ber(res.ber_closed)}  (AWGN closed form)\n"
        f"  link {'ON AIR' if res.link_ok else 'OUT'}"
        f"   payload this frame "
        f"{_fmt_bits(res.n_bits if res.link_ok else 0)}{switch}\n"
        f"  on-air {sim.on_air}/{sim.frame} frames "
        f"({sim.availability:.1f}%)  "
        f"cumulative {_fmt_bits(sim.payload_bits)}"
    )


def run_headless(n_frames, seed=SEED_DEFAULT, bits=8192):
    """Print-only ramp: distance 1 -> 15 over n_frames, demo.py-style table."""
    if n_frames < 2:
        print("--headless needs at least 2 frames", file=sys.stderr)
        return 2
    sim = LinkSim(seed=seed)
    print(f"interactive demo (headless) | {n_frames} frames | "
          f"Rs = {SYMBOL_RATE:.0e} symb/s | "
          f"hysteresis {HYSTERESIS_GAP_DB:.1f} dB | wobble on")
    header = (f"{'frame':>5}  {'dist':>5}  {'true':>6}  {'est':>6}  "
              f"{'mode':>6}  {'BER':>8}  {'link':>4}  {'next':>6}  "
              f"{'payload':>9}  {'cum':>10}")
    print(header)
    print("-" * len(header))
    distances = np.linspace(1.0, 15.0, n_frames)
    for dist in distances:
        cfg = FrameConfig(distance=float(dist), wobble=True,
                          frame_bits=bits)
        res = sim.step(cfg)
        payload = res.n_bits if res.link_ok else 0
        print(f"{res.frame + 1:>5}  {dist:>5.1f}  {res.es_true_db:>6.2f}  "
              f"{res.es_est_db:>6.2f}  {res.mode_tx:>6}  "
              f"{res.ber_measured:>8.1e}  "
              f"{'OK' if res.link_ok else 'OUT':>4}"
              f"  {res.mode_next:>6}  {payload:>9}  "
              f"{sim.payload_bits:>10}")
    print(f"\navailability: {sim.on_air}/{n_frames} dwells "
          f"({sim.availability:.1f}%)")
    print(f"cumulative payload on air: {_fmt_bits(sim.payload_bits)}")
    print(f"switches: {len(sim.switches)}   "
          f"last mode: {sim.selector.mode}")
    return 0


def run_gui(seed=SEED_DEFAULT, bits=8192, selftest=False):
    """Build and run the Tkinter demo; returns a process exit code."""
    import tkinter as tk
    from tkinter import ttk

    # Figure + FigureCanvasTkAgg need no pyplot, but pinning the backend
    # before any figure exists keeps rendering on the Tk canvas.
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"cannot open a display ({exc})\n"
              f"use: .venv/bin/python interactive_demo.py --headless 40",
              file=sys.stderr)
        return 1

    root.title("Adaptive modulation - live link demo")
    app = _DemoApp(root, tk, ttk, Figure, FigureCanvasTkAgg,
                   seed=seed, frame_bits=bits, selftest=selftest)
    root.mainloop()
    if selftest:
        print(f"selftest: OK ({app.sim.frame} frames rendered)")
    return 0


class _DemoApp:
    """Widget tree + redraw logic. Built by run_gui() so the module-level
    import stays tkinter-free for the unit tests."""

    def __init__(self, root, tk, ttk, Figure, FigureCanvasTkAgg,
                 seed=SEED_DEFAULT, frame_bits=8192, selftest=False):
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.sim = LinkSim(seed=seed)
        self.auto_on = False
        self._build(Figure, FigureCanvasTkAgg, frame_bits)
        self.step_once()
        if selftest:
            for delay in (300, 600, 900):
                root.after(delay, self.step_once)
            root.after(1800, self._finish_selftest)

    # ---------------------------------------------------------------- UI --
    def _build(self, Figure, FigureCanvasTkAgg, frame_bits):
        tk, ttk = self.tk, self.ttk

        self.banner = tk.Label(
            self.root, text="", font=("TkDefaultFont", 15, "bold"),
            fg="white", bg=COLORS["bpsk"], pady=6)
        self.banner.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.subbanner = tk.Label(self.root, text="",
                                  font=("TkDefaultFont", 9))
        self.subbanner.grid(row=1, column=0, columnspan=3, sticky="w",
                            padx=8)

        # --- channel controls -------------------------------------------
        ctrl = ttk.LabelFrame(self.root, text="Channel controls", padding=8)
        ctrl.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)

        ttk.Label(ctrl, text="distance (units)").grid(row=0, column=0,
                                                      sticky="w")
        self.dist_var = tk.DoubleVar(value=1.0)
        self.dist_scale = tk.Scale(
            ctrl, from_=1.0, to=15.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=230, showvalue=True,
            variable=self.dist_var, command=lambda _v: self._readouts())
        self.dist_scale.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.es_label = ttk.Label(ctrl, text="", font=("TkFixedFont", 10))
        self.es_label.grid(row=2, column=0, columnspan=2, sticky="w",
                           pady=(4, 2))

        self.fading_var = tk.StringVar(value="awgn")
        ttk.Label(ctrl, text="channel:").grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(ctrl, text="AWGN", value="awgn",
                        variable=self.fading_var).grid(row=3, column=1,
                                                       sticky="w")
        ttk.Radiobutton(ctrl, text="Rayleigh fade", value="rayleigh",
                        variable=self.fading_var).grid(row=4, column=1,
                                                       sticky="w")

        self.wobble_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="set-point wobble (+-0.4 dB)",
                        variable=self.wobble_var).grid(row=5, column=0,
                                                       columnspan=2,
                                                       sticky="w")

        ttk.Label(ctrl, text="path-loss exponent").grid(row=6, column=0,
                                                        sticky="w")
        self.exp_var = tk.DoubleVar(value=2.2)
        tk.Scale(ctrl, from_=2.0, to=4.0, resolution=0.1,
                 orient=tk.HORIZONTAL, length=230, showvalue=True,
                 variable=self.exp_var,
                 command=lambda _v: self._readouts()).grid(
            row=7, column=0, columnspan=2, sticky="ew")

        ttk.Label(ctrl, text="frame size (bits)").grid(row=8, column=0,
                                                       sticky="w")
        self.bits_var = tk.StringVar(value=str(frame_bits))
        ttk.Combobox(ctrl, textvariable=self.bits_var, state="readonly",
                     values=("2048", "8192", "32768"), width=8).grid(
            row=9, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(ctrl)
        btns.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Step", command=self.step_once).pack(
            side=tk.LEFT, padx=2)
        self.auto_btn = ttk.Button(btns, text="Auto",
                                   command=self.toggle_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Reset", command=self.reset).pack(
            side=tk.LEFT, padx=2)
        ctrl.columnconfigure(0, weight=1)

        # --- TX/RX result panels ----------------------------------------
        results = ttk.LabelFrame(self.root, text="TX / RX results",
                                 padding=4)
        results.grid(row=2, column=1, sticky="nsew", padx=6, pady=4)
        self.results_text = tk.Text(
            results, width=52, height=19, font=("TkFixedFont", 9),
            state="disabled", relief="flat",
            background=self.root.cget("background"))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # --- constellation canvas ---------------------------------------
        const_frame = ttk.LabelFrame(self.root, text="Constellations",
                                     padding=4)
        const_frame.grid(row=2, column=2, sticky="nsew", padx=6, pady=4)
        self.const_fig = Figure(figsize=(5.6, 3.4), dpi=100)
        self.ax_tx = self.const_fig.add_subplot(1, 2, 1)
        self.ax_rx = self.const_fig.add_subplot(1, 2, 2)
        self.const_canvas = FigureCanvasTkAgg(self.const_fig,
                                              master=const_frame)
        self.const_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- switch log ---------------------------------------------------
        log_frame = ttk.LabelFrame(self.root,
                                   text="Switch log (newest first)",
                                   padding=4)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=4)
        self.log_list = tk.Listbox(log_frame, height=7,
                                   font=("TkFixedFont", 9),
                                   exportselection=False)
        self.log_list.pack(fill=tk.BOTH, expand=True)

        # --- trace canvas -------------------------------------------------
        trace_frame = ttk.LabelFrame(self.root,
                                     text="Es/N0 trace + thresholds",
                                     padding=4)
        trace_frame.grid(row=3, column=1, columnspan=2, sticky="nsew",
                         padx=6, pady=4)
        self.trace_fig = Figure(figsize=(8.0, 2.6), dpi=100)
        self.ax_trace = self.trace_fig.add_subplot(1, 1, 1)
        self.trace_canvas = FigureCanvasTkAgg(self.trace_fig,
                                              master=trace_frame)
        self.trace_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)

    # ------------------------------------------------------------ config --
    def _cfg(self):
        return FrameConfig(
            distance=float(self.dist_var.get()),
            fading=self.fading_var.get() == "rayleigh",
            wobble=bool(self.wobble_var.get()),
            frame_bits=int(self.bits_var.get()),
            exponent=float(self.exp_var.get()),
        )

    def _readouts(self):
        cfg = self._cfg()
        self.es_label.config(
            text=f"set-point Es/N0 = {set_point_es_n0(cfg):.1f} dB")

    # ------------------------------------------------------------ steps ---
    def step_once(self):
        cfg = self._cfg()
        res = self.sim.step(cfg)
        self._redraw(res, cfg)

    def toggle_auto(self):
        self.auto_on = not self.auto_on
        self.auto_btn.config(text="Stop" if self.auto_on else "Auto")
        if self.auto_on:
            if self.dist_var.get() >= 15.0 - 1e-9:
                self.dist_var.set(1.0)
                self._readouts()
            self.root.after(AUTO_MS, self._auto_tick)

    def _auto_tick(self):
        if not self.auto_on:
            return
        dist = float(self.dist_var.get())
        if dist >= 15.0 - 1e-9:
            self.auto_on = False
            self.auto_btn.config(text="Auto")
            return
        self.dist_var.set(min(15.0, dist + AUTO_STEP))
        self._readouts()
        self.step_once()
        self.root.after(AUTO_MS, self._auto_tick)

    def reset(self):
        self.auto_on = False
        self.auto_btn.config(text="Auto")
        self.sim.reset()
        self.dist_var.set(1.0)
        self.exp_var.set(2.2)
        self.fading_var.set("awgn")
        self.wobble_var.set(True)
        self.log_list.delete(0, self.tk.END)
        self._readouts()
        self.step_once()

    def _finish_selftest(self):
        self.root.destroy()

    # ------------------------------------------------------------ redraw --
    def _redraw(self, res, cfg):
        tk = self.tk
        mode = res.mode_tx
        state = "ON AIR" if res.link_ok else "LINK OUT"
        self.banner.config(
            text=f"{mode.upper()}  k={res.k}   {state}   "
                 f"{_fmt_rate(res.throughput_bps)}",
            bg=COLORS[mode], fg=BANNER_FG.get(mode, "white"))
        swapped = ("" if res.mode_next == mode else
                   f"   next frame -> {res.mode_next}")
        self.subbanner.config(
            text=f"frame #{res.frame}  mode chosen from frame "
                 f"#{res.frame - 1} estimate (one-dwell latency)"
                 f"{swapped}")

        body = _results_text(res, self.sim, cfg)
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", body)
        self.results_text.config(state="disabled")

        if res.mode_next != mode:
            self.log_list.insert(
                0, f"#{res.frame}: {mode} -> {res.mode_next}  "
                   f"@ est {res.es_est_db:.1f} dB")
            while self.log_list.size() > 8:
                self.log_list.delete(tk.END)

        self._draw_constellations(res)
        self._draw_trace()

    def _draw_constellations(self, res):
        mode = res.mode_tx
        color = COLORS[mode]

        self.ax_tx.cla()
        tx = res.tx_syms
        step = max(1, tx.size // 600)
        self.ax_tx.scatter(tx[::step].real, tx[::step].imag, s=8,
                           c=color, alpha=0.55, linewidths=0)
        self.ax_tx.set_title(f"TX data ({mode})", fontsize=9)

        self.ax_rx.cla()
        rx = res.rx_syms
        step = max(1, rx.size // 600)
        self.ax_rx.scatter(rx[::step].real, rx[::step].imag, s=8,
                           c=color, alpha=0.45, linewidths=0)
        err_idx = np.flatnonzero(res.sym_err_mask)
        if err_idx.size:
            estep = max(1, err_idx.size // 400)
            e = err_idx[::estep]
            self.ax_rx.scatter(rx[e].real, rx[e].imag, s=22, marker="x",
                               c="#000000", linewidths=1.2)
        self.ax_rx.set_title(
            f"RX at decisions  {res.n_err}/{res.n_bits} bits wrong",
            fontsize=9)

        for ax in (self.ax_tx, self.ax_rx):
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        self.const_fig.tight_layout()
        self.const_canvas.draw_idle()

    def _draw_trace(self):
        ax = self.ax_trace
        ax.cla()
        hist = self.sim.history
        frames = [r.frame for r in hist]
        for r in hist:
            ax.axvspan(r.frame - 0.5, r.frame + 0.5,
                       color=COLORS[r.mode_tx], alpha=0.22, linewidth=0)
        ax.plot(frames, [r.es_true_db for r in hist], color="#333333",
                lw=1.2, label="true Es/N0")
        ax.plot(frames, [r.es_est_db for r in hist], color="#1f77b4",
                lw=1.0, marker="o", ms=2.5, label="pilot estimate")
        for scheme in ALL_SCHEMES:
            up = THRESHOLDS_ES_N0_DB[scheme] + HYSTERESIS_GAP_DB / 2.0
            ax.axhline(up, color=COLORS[scheme], ls=":", lw=1.0,
                       label=f"{scheme} up-shift {up:.2f}")
        ax.axvline(hist[-1].frame, color="#000000", lw=0.8, alpha=0.4)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("Es/N0 (dB)", fontsize=8)
        ax.set_xlim(-0.5, max(6, frames[-1] + 0.5))
        all_vals = ([r.es_true_db for r in hist]
                    + [r.es_est_db for r in hist]
                    + [THRESHOLDS_ES_N0_DB[s] + HYSTERESIS_GAP_DB / 2.0
                       for s in ALL_SCHEMES])
        ax.set_ylim(min(all_vals) - 1.0, max(max(all_vals) + 1.0, 28.0))
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower left", ncol=3, framealpha=0.7)
        self.trace_fig.tight_layout()
        self.trace_canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive adaptive-modulation demo: vary the channel "
                    "and watch the link pick the best modulation.")
    parser.add_argument("--headless", type=int, metavar="N", default=None,
                        help="run N frames over distance 1..15 without a "
                             "GUI and print the dwell table")
    parser.add_argument("--selftest", action="store_true",
                        help="open the GUI, render a few frames, close "
                             "after ~2 s (smoke test, needs a display)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--bits", type=int, default=8192,
                        choices=(2048, 8192, 32768),
                        help="frame payload size (default 8192)")
    args = parser.parse_args(argv)

    if args.headless is not None:
        return run_headless(args.headless, seed=args.seed, bits=args.bits)
    return run_gui(seed=args.seed, bits=args.bits, selftest=args.selftest)


if __name__ == "__main__":
    raise SystemExit(main())
