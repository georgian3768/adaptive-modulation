# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Simulation driver: Monte Carlo loops, adaptive-vs-fixed comparison, plotting.

Implemented so far:
    * BPSK over AWGN: simulated BER vs theory          -> results/bpsk_check.png
    * All four schemes over AWGN: sim vs theory BER    -> results/ber_all_schemes.png
    * Pulse-shaping regression: the pulse-shaped chain vs the symbol-level
      chain must give the SAME BER under ideal sync; max deviation is printed.
    * Adaptive-vs-fixed: adaptive link (adapt.py thresholds + hysteresis)
      against BPSK/QPSK/16-QAM/64-QAM across a receding-unit SNR trace,
      per-dwell BER and throughput -> results/adaptive_vs_fixed.png
      (python main.py --adaptive-vs-fixed).

Per the project's verification rule, no simulated BER curve is trusted until it
is compared against its closed-form expression.

Theoretical BERs used below (closed forms):
    BPSK / QPSK:   Q(sqrt(2*Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))
                   (identical for Gray-coded QPSK: one bit per axis, same as
                   two independent BPSK paths)
    16-QAM / 64-QAM (approximate, excellent at BER below ~1e-2):
                   (4/k)*(1 - 1/sqrt(M)) * Q(sqrt(3*k*Eb/N0/(M-1)))
                   = coef * 0.5*erfc(sqrt(3*k*Eb/N0/(2*(M-1))))
    where k = log2(M) = bits/symbol and M = 16 or 64. This is the standard
    textbook formula. It drops with two approximations: the QAM grid is
    projected onto its two PAM axes (leading symbol-error term
    4(1-1/sqrt(M)) Q(...) dominates, BER = P_s/k), and every symbol error is
    counted as one Gray-code bit flip - both accurate at low BER.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend - the sim runs on Debian without a display
import matplotlib.pyplot as plt
from scipy.special import erfc

from modem import BITS_PER_SYMBOL, modulate, demodulate, constellation_points
from channel import (add_awgn, add_awgn_samples, apply_flat_fading,
                     noise_power_from_eb_n0)
from pulse_shape import rrc_filter, tx_pulse_shape, rx_match_filter

ALL_SCHEMES = ["bpsk", "qpsk", "16qam", "64qam"]
COLORS = {"bpsk": "#1f77b4", "qpsk": "#ff7f0e", "16qam": "#2ca02c", "64qam": "#d62728"}
MARKERS = {"bpsk": "o", "qpsk": "s", "16qam": "^", "64qam": "d"}


def theoretical_ber(scheme, eb_n0_db):
    """Closed-form BER for one scheme as a function of Eb/N0 in dB."""
    eb_n0_lin = 10.0 ** (np.asarray(eb_n0_db) / 10.0)
    k = BITS_PER_SYMBOL[scheme]
    if scheme in ("bpsk", "qpsk"):
        return 0.5 * erfc(np.sqrt(eb_n0_lin))
    m = 2 ** k
    # Symbol-error leading term 4(1-1/sqrt(M)) Q(...), divided by k bits.
    coef = (4.0 / k) * (1.0 - 1.0 / np.sqrt(m))
    return coef * 0.5 * erfc(np.sqrt(3.0 * k * eb_n0_lin / (2.0 * (m - 1.0))))


def monte_carlo_ber(scheme, eb_n0_db, min_errors=30, max_bits=2_000_000,
                    seed=42, faded=False):
    """Estimate BER by injecting bits until enough errors are counted.

    Instead of a fixed number of bits per point, we keep drawing blocks until
    either `min_errors` bit errors have been seen (gives ~15% relative error
    at the noisy-SNR points) or `max_bits` is exhausted (very clean channels
    where the true BER is far below 1/max_bits -> est. 0). This keeps every
    plotted point statistically meaningful without wasting time drawing
    millions of bits in the region where the scheme is almost error-free.
    The same seed makes every run reproducible.

    faded=True applies channel.apply_flat_fading() before the AWGN: an
    independent per-symbol Rayleigh amplitude (E[alpha^2]=1). Noise is set
    from the UNFADED symbol energy, so Eb/N0 is the standard average Eb/N0
    under fading (see channel.apply_flat_fading for why per-symbol draws,
    which give the same averaged BER as slow block fading here).
    """
    k = BITS_PER_SYMBOL[scheme]
    batch = 240_000                # divisible by 1,2,4,6 -> no partial symbols
    rng = np.random.default_rng(seed)
    total_err = 0
    total_bits = 0
    while total_bits < max_bits and total_err < min_errors:
        bits = rng.integers(0, 2, batch)
        symbols = modulate(bits, scheme)
        if faded:
            # Keep N0 referenced to the launching energy: a |g|^2-scaled Es
            # measured from the faded data would cancel the fading entirely.
            es_nominal = float(np.mean(np.abs(symbols) ** 2))
            faded, alpha = apply_flat_fading(symbols, rng=rng)
            rx_symbols, _ = add_awgn(faded, eb_n0_db, bits_per_symbol=k,
                                     rng=rng, symbol_energy=es_nominal)
            # Ideal AGC: undo the fade at the receiver; the decision lattice
            # is fixed at the unit-energy constellation (see channel docstring).
            rx_symbols = rx_symbols / alpha
        else:
            rx_symbols, _ = add_awgn(symbols, eb_n0_db, bits_per_symbol=k, rng=rng)
        rx_bits = demodulate(rx_symbols, scheme)
        n_err = int(np.count_nonzero(bits != rx_bits))
        total_err += n_err
        total_bits += batch
    return total_err / total_bits


def monte_carlo_ber_shaped(scheme, eb_n0_db, rolloff=0.22, span=10, sps=8,
                           min_errors=30, max_bits=2_000_000, seed=42,
                           faded=False):
    """BER with the full pulse-shaped chain.

    bits -> modulate -> upsample -> RRC TX filter -> AWGN at the sample rate
         -> RRC matched filter -> downsample at symbol centres -> demodulate

    faded=True applies channel.apply_flat_fading() to the SYMBOLS BEFORE the
    pulse shaper (per-symbol Rayleigh amplitude). The TX/RX filters are
    linear, so the fade multiplies the decision-point symbols the same way it
    does in the symbol-level chain; both chains see identical per-symbol
    alphas and their faded curves will agree within Monte Carlo noise. Noise
    N0 stays referenced to the unfaded symbol energy (average-Eb/N0
    convention).

    The same statistical protocol (min_errors / max_bits / seed) is used as in
    monte_carlo_ber() so the two runs are directly comparable. Noise is added
    at the SAMPLE rate with total power N0 (N0/2 per dimension per sample) via
    channel.add_awgn_samples(); because both filters are unit-norm, the noise
    at the decision point is the same N0/2 per dimension as in the
    symbol-level chain, so the
    shaped curve must land on top of the symbol-level curve (the regression
    check below confirms it).
    """
    k = BITS_PER_SYMBOL[scheme]
    batch = 240_000                # divisible by 1,2,4,6 -> no partial symbols
    rng = np.random.default_rng(seed)

    h = rrc_filter(rolloff, span, sps)
    # N0 referenced to the ORIGINAL symbol energy Es = 1 (not to the upsampled
    # waveform, whose average power is Es/sps).
    n0 = noise_power_from_eb_n0(eb_n0_db, k, symbol_energy=1.0)
    trim = span // 2               # drop edge-transient symbols each side

    total_err = 0
    total_bits = 0
    while total_bits < max_bits and total_err < min_errors:
        bits = rng.integers(0, 2, batch)
        symbols = modulate(bits, scheme)
        if faded:
            symbols, alpha = apply_flat_fading(symbols, rng=rng)
        n_sym = symbols.size
        wave = tx_pulse_shape(symbols, h, sps)
        wave = add_awgn_samples(wave, n0, rng=rng)
        rx_symbols = rx_match_filter(wave, h, sps, n_sym)[trim:-trim]
        if faded:
            # Ideal AGC: undo the per-symbol fade before hard decisions. The
            # RC filter leaves symbol n's own alpha_n intact at its centre.
            rx_symbols = rx_symbols / alpha[trim:-trim]

        rx_bits = demodulate(rx_symbols, scheme)
        ref = bits[trim * k: (n_sym - trim) * k]   # exclude the trimmed edges
        n_err = int(np.count_nonzero(ref != rx_bits))
        total_err += n_err
        total_bits += ref.size
    return total_err / total_bits


def _print_comparison(snr_db, sim_ber, theo_ber, scheme, sim2=None):
    """Terminal check: simulated vs theoretical BER for one scheme.

    sim2 (optional) is a second simulation array (the pulse-shaped chain);
    adding it lets the regression detail be read point-by-point.
    """
    print(f"--- {scheme}: simulated vs theory ---")
    if sim2 is None:
        print(f"{'Eb/N0 dB':>8} {'sim BER':>12} {'theory':>12} {'sim/th':>8}")
        for e, s_ber, t_ber in zip(snr_db, sim_ber, theo_ber):
            ratio = s_ber / t_ber if t_ber > 0 else float("nan")
            print(f"{e:8.1f} {s_ber:12.5g} {t_ber:12.5g} {ratio:8.3f}")
    else:
        print(f"{'Eb/N0 dB':>8} {'Sym BER':>12} {'Shaped BER':>12} {'sim/th':>8} {'Sha/Sym':>8}")
        for e, s_ber, s2, t_ber in zip(snr_db, sim_ber, sim2, theo_ber):
            ratio = s_ber / t_ber if t_ber > 0 else float("nan")
            sha_sym = s2 / s_ber if s_ber > 0 else float("nan")
            print(f"{e:8.1f} {s_ber:12.5g} {s2:12.5g} {ratio:8.3f} {sha_sym:8.3f}")


def run_bpsk_check():
    """BPSK-only check (0..12 dB), saved to results/bpsk_check.png."""
    snr_db = np.arange(0.0, 13.0, 1.0)
    sim_ber = np.array([monte_carlo_ber("bpsk", e, seed=42) for e in snr_db])
    theo_ber = theoretical_ber("bpsk", snr_db)

    _print_comparison(snr_db, sim_ber, theo_ber, "bpsk")

    plt.figure(figsize=(7, 5))
    plt.semilogy(snr_db, theo_ber, "b-", linewidth=1.5,
                 label="Theory $Q(\\sqrt{2E_b/N_0})$")
    ok = sim_ber > 0   # BER = 0 points cannot be drawn on a log axis
    plt.semilogy(snr_db[ok], sim_ber[ok], "ro", markersize=5, label="Simulated")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BPSK over AWGN: simulation vs theory")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = "results/bpsk_check.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def run_all_schemes_check():
    """Pulse-shaping regression + plot: shaped vs symbol-level vs theory, 0..24 dB.

    Runs the SAME Monte Carlo sweep twice per scheme - once on the
    symbol-level chain and once on the pulse-shaped chain (same seeds,
    same min-errors protocol) - prints the max deviation between the two as
    the regression test, then draws all curves over the theory on one figure.

    Returns {scheme: (sym_ber, shaped_ber, theo_ber)} so the caller can run
    further verification (verify_shaped_regression) without re-running the
    sweeps.
    """
    snr_db = np.arange(0.0, 25.0, 1.0)
    print("\n=== Pulse-shaped vs symbol-level regression ===")
    plt.figure(figsize=(8, 6))
    results = {}

    for scheme in ALL_SCHEMES:
        theo_ber = theoretical_ber(scheme, snr_db)
        sym_ber = np.array([monte_carlo_ber(scheme, e, seed=42) for e in snr_db])
        shaped_ber = np.array([monte_carlo_ber_shaped(scheme, e, seed=42) for e in snr_db])

        # Regression check: symbol-level reference bits vs shaped-chain bits.
        abs_dev = np.abs(shaped_ber - sym_ber)
        imax = int(np.argmax(abs_dev))
        # Relative deviation measured only where the estimate is statistically
        # solid (>= 100 expected errors). Below ~1e-4 the min-errors cap makes
        # both runs count single-digit errors, so ratios there are pure noise.
        strong = sym_ber >= 1e-4
        rel_strong = (abs_dev[strong] / sym_ber[strong]).max() if strong.any() else 0.0
        zero_mismatch = int(np.sum((sym_ber == 0) != (shaped_ber == 0)))
        print(f"  {scheme:6s}: max |sym - shaped| = {abs_dev[imax]:.3e} "
              f"at {snr_db[imax]:.0f} dB")
        print(f"           max rel dev (BER >= 1e-4) = {rel_strong:.3f}   "
              f"floor mismatches (0 vs nonzero) = {zero_mismatch}")

        _print_comparison(snr_db, sym_ber, theo_ber, scheme, sim2=shaped_ber)

        color, marker = COLORS[scheme], MARKERS[scheme]
        plt.semilogy(snr_db, theo_ber, color=color, linestyle="-",
                     linewidth=1.2, label=f"{scheme} (theory)")
        ok_sym = sym_ber > 0
        ok_shaped = shaped_ber > 0
        plt.semilogy(snr_db[ok_sym], sym_ber[ok_sym], color=color, marker=marker,
                     linestyle="", markersize=4, label=f"{scheme} (symbol-level)")
        plt.semilogy(snr_db[ok_shaped], shaped_ber[ok_shaped], color=color, marker="x",
                     linestyle="", markersize=5, label=f"{scheme} (pulse-shaped)")

        results[scheme] = (sym_ber, shaped_ber, theo_ber)

    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs $E_b/N_0$: pulse-shaped vs symbol-level chain")
    plt.xlim(0, 24)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    out = "results/ber_all_schemes.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    return results


# Shaped-chain verification pass/fail tolerances: max absolute BER deviation
# anywhere (Monte Carlo scatter at high-BER points dominates this) and max
# relative deviation where the estimate is statistically solid (BER >= 1e-4).
# Both were observed to stay under these bounds with span=10; with span=8 the
# 64-QAM count exceeded them (systematic ISI degradation).
MAX_ABS_DEV_TOL = 5e-3
MAX_REL_DEV_TOL = 0.5


def verify_shaped_regression(results):
    """Print the pulse-shaping regression verdict and two analog checks.

    Uses the sweeps already run by run_all_schemes_check() (passed in as
    results) so nothing is re-simulated. Three checks:

      1. BER regression: max |sym - shaped| per scheme, PASSed against a
         Monte-Carlo-noise bound rather than a hand wave.
      2. Combined filter response: TX RRC convolved with the RX matched RRC
         must be Nyquist (zero crossings at every nonzero integer symbol
         interval). Drawn to results/rc_impulse_response.png.
      3. Eye diagram: matched-filter output overlaid across many symbol
         windows, eyes clearly open at the sampling instant. Saved to
         results/eye_diagram.png.
    """
    snr_db = np.arange(0.0, 25.0, 1.0)
    print("\n=== Pulse-shaped vs symbol-level verification ===")

    all_ok = True
    for scheme in ALL_SCHEMES:
        sym_ber, shaped_ber, _ = results[scheme]
        dev = np.abs(shaped_ber - sym_ber)
        imax = int(np.argmax(dev))
        strong = sym_ber >= 1e-4
        rel_best = (dev[strong] / sym_ber[strong]).max() if strong.any() else 0.0
        ok = dev[imax] <= MAX_ABS_DEV_TOL and rel_best <= MAX_REL_DEV_TOL
        all_ok &= ok
        print(f"  {scheme:6s}: max |sym - shaped| = {dev[imax]:.3e} "
              f"(at {snr_db[imax]:.0f} dB)  "
              f"max rel dev (BER>=1e-4) = {rel_best:.3f}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'} "
          f"(tolerance |dev| <= {MAX_ABS_DEV_TOL:.0e}, "
          f"rel <= {MAX_REL_DEV_TOL:.1f})")

    _plot_combined_response()
    _plot_eye_diagram()
    return all_ok


def _plot_combined_response(rolloff=0.22, span=10, sps=8):
    """RRC(TX) convolved with RRC(RX) is the raised-cosine pulse: show it.

    Prints the residual response at every nonzero integer symbol offset (must
    be ~1e-3, i.e. the finite-span truncation leftover) and draws the pulse
    with the symbol grid overlaid so the zero crossings are visible.
    """
    h = rrc_filter(rolloff, span, sps)
    rc = np.convolve(h, h)                      # combined TX+RX response
    centre = rc.size // 2
    t = (np.arange(rc.size) - centre) / sps     # x axis in symbol periods

    lags = np.arange(1, span // 2 + 1)
    crossings = np.array([rc[centre + m * sps] for m in lags])
    worst = float(np.abs(crossings).max())
    print(f"  RRC*RRC Nyquist check: max |combined response| at nonzero "
          f"integer symbol offsets = {worst:.3e} (finite-span residual, "
          f"theory: 0)")
    for m, v in zip(lags, crossings):
        print(f"      {m:+d} T: {v:+.3e}")

    plt.figure(figsize=(8, 5))
    plt.plot(t, rc, linewidth=1.5, label="RRC * RRC (raised cosine)")
    plt.axhline(0, color="gray", linewidth=0.6)
    grid = np.arange(-(span // 2), span // 2 + 1)
    for m in grid:
        plt.axvline(m, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
    for m in lags:
        plt.plot(m, rc[centre + m * sps], "ko", markersize=4, zorder=5)
    plt.plot(0, rc[centre], "ko", markersize=6, zorder=5)
    plt.xlabel("time (symbol periods T)")
    plt.ylabel("combined response")
    plt.title(f"TX RRC $\\oplus$ RX RRC (rolloff={rolloff}, span={span}): "
              "zero crossings at integer T")
    plt.xlim(-span / 2, span / 2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = "results/rc_impulse_response.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")


def _eye_opening(re_iso, levels):
    """Vertical eye margin at the sampling instant, in constellation units.

    re_iso: real (I) part of the matched-filter output sampled at the symbol
    centres. levels: the scheme's I-axis decision levels. Each sample is
    assigned to its nearest level; opening = min over adjacent-level pairs of
    (half-gap - 3*sigma), i.e. the worst point where a 3-sigma noise cloud
    around a level stays clear of the decision boundary. Positive -> open eye.
    """
    levels = np.unique(np.asarray(levels))
    idx = np.argmin(np.abs(re_iso[:, None] - levels[None, :]), axis=1)
    sigma = np.array([re_iso[idx == i].std() for i in range(levels.size)])
    half_gap = np.diff(levels) / 2.0
    worst_noise = np.maximum(sigma[:-1], sigma[1:])
    margin = float((half_gap - 3.0 * worst_noise).min())
    return margin, sigma


def _plot_eye_diagram(rolloff=0.22, span=10, sps=8, n_symbols=8000,
                      n_traces=400, seed=0):
    """Eye diagram of the matched-filter output (I channel) for two schemes.

    Traces are one-symbol-period segments centred on the symbol sampling
    instants, overlaid; a red line marks the decision point at t = 0. Open
    eyes at that instant are the visible symptom of zero ISI. The numeric
    vertical margin (see _eye_opening) is printed so "open" is a measured
    fact, not a judgement call. Noise is added at a realistic operating point
    (QPSK @ 8 dB, 16-QAM @ 12 dB) so the eye opening has a natural width.
    """
    h = rrc_filter(rolloff, span, sps)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (scheme, eb_db) in zip(axes, (("qpsk", 8), ("16qam", 12))):
        k = BITS_PER_SYMBOL[scheme]
        n_bits = n_symbols * k
        bits = rng.integers(0, 2, n_bits)
        symbols = modulate(bits, scheme)
        wave = tx_pulse_shape(symbols, h, sps)
        n0 = noise_power_from_eb_n0(eb_db, k, symbol_energy=1.0)
        wave = add_awgn_samples(wave, n0, rng=rng)
        y = np.real(np.convolve(wave, h))       # matched-filter output, I axis

        centre = (len(h) - 1) + span * sps      # first edge-free symbol centre
        iso = y[centre::sps][: n_symbols - 2 * span]
        margin, sigma = _eye_opening(
            iso, np.unique(constellation_points(scheme).real))
        state = "open" if margin > 0 else "CLOSED"
        print(f"  {scheme:6s} eye @ {eb_db} dB: I-axis noise sigma(s) = "
              f"{np.round(sigma, 4)}; vertical margin (min half-gap - 3 sigma) "
              f"= {margin:+.4f} -> {state}")

        for n in rng.choice(np.arange(span, n_symbols - span), size=n_traces,
                            replace=False):
            idx0 = (len(h) - 1) + (n - 1) * sps   # symbol n centre = sps samples in
            seg = y[idx0: idx0 + 2 * sps]
            tt = np.linspace(-1, 1, seg.size)
            ax.plot(tt, seg, color="C0", linewidth=0.4, alpha=0.15)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.9,
                   label="sampling instant")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{scheme} eye diagram  (I channel, $E_b/N_0$ = "
                     f"{eb_db} dB, {n_traces} traces)")
        ax.set_xlabel(r"$t - t_s$ / T")
        ax.set_ylabel("matched-filter output")
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = "results/eye_diagram.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")


def _snr_at_ber(snr_db, ber, target):
    """Eb/N0 (dB) where a simulated BER curve crosses `target`, log-interpolated.

    Returns None when the curve stays above target across the whole range
    (the required Eb/N0 is off the right of the sweep).
    """
    ber = np.asarray(ber)
    hits = ber < target
    if not hits.any():
        return None
    i_hi = int(np.argmax(hits))
    if i_hi < 1:
        return float(snr_db[0])
    i_lo = i_hi - 1
    b_lo, b_hi = ber[i_lo], ber[i_hi]
    if b_lo <= 0 or b_hi <= 0 or b_lo == b_hi:
        return None
    x_lo, x_hi = float(snr_db[i_lo]), float(snr_db[i_hi])
    t = np.log(target / b_lo) / np.log(b_hi / b_lo)
    return float(x_lo + t * (x_hi - x_lo))


def run_fading_comparison(results, target_ber=1e-2):
    """Fading-penalty sweep, run with: python main.py --fading

    Re-runs the symbol-level and pulse-shaped BER sweeps with flat Rayleigh
    fading enabled (channel.apply_flat_fading before the AWGN, same
    seeds/min-errors protocol) and plots, per modulation scheme on one figure,
    the AWGN-only curves against the faded ones - symbol-level AND
    pulse-shaped - so the fading penalty is visible. Also prints the extra
    Eb/N0 needed to reach target_ber under fading (log-interpolated).

    Saves results/fading_vs_awgn.png. The AWGN-only curves come from the
    results already computed by run_all_schemes_check(); only the faded
    sweeps are re-simulated.
    """
    snr_db = np.arange(0.0, 25.0, 1.0)
    print("\n=== Fading penalty: flat Rayleigh vs AWGN-only (avg Eb/N0) ===")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, scheme in zip(axes, ALL_SCHEMES):
        awgn_ber, _, _ = results[scheme]     # AWGN-only, already simulated
        faded_ber = np.array([monte_carlo_ber(scheme, e, seed=42, faded=True)
                              for e in snr_db])
        faded_shaped = np.array([monte_carlo_ber_shaped(scheme, e, seed=42,
                                                        faded=True) for e in snr_db])

        snr_awgn = _snr_at_ber(snr_db, awgn_ber, target_ber)
        snr_fade = _snr_at_ber(snr_db, faded_ber, target_ber)
        if snr_awgn is not None and snr_fade is not None:
            print(f"  {scheme:6s}: Eb/N0 @ BER={target_ber:.0e}: "
                  f"AWGN {snr_awgn:5.2f} dB, faded {snr_fade:5.2f} dB  "
                  f"-> +{snr_fade - snr_awgn:5.2f} dB penalty")
        elif snr_awgn is not None:
            print(f"  {scheme:6s}: BER={target_ber:.0e} never reached under "
                  f"fading by {snr_db.max():.0f} dB (AWGN needs "
                  f"{snr_awgn:.2f} dB)")

        color, marker = COLORS[scheme], MARKERS[scheme]
        ok0 = awgn_ber > 0
        ok1 = faded_ber > 0
        ok2 = faded_shaped > 0
        ax.semilogy(snr_db[ok0], awgn_ber[ok0], color=color, marker=marker,
                    linestyle="-", linewidth=1.1, markersize=4,
                    label="AWGN")
        ax.semilogy(snr_db[ok1], faded_ber[ok1], color=color, linestyle="--",
                    linewidth=1.5, label="Rayleigh fading (symbol-level)")
        ax.semilogy(snr_db[ok2], faded_shaped[ok2], color=color, linestyle=":",
                    linewidth=1.5, label="Rayleigh fading (pulse-shaped)")
        ax.set_title(scheme)
        ax.set_xlim(0, 24)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylabel("BER")

    fig.suptitle("Flat Rayleigh fading vs AWGN (average $E_b/N_0$)")
    axes[-2].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[-1].set_xlabel(r"$E_b/N_0$ (dB)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = "results/fading_vs_awgn.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def synthetic_snr_trace(n_dwells=40, snr0_db=27.0, d_far=15.0, exponent=2.2,
                        wobble_db=0.4, seed=7):
    """Es/N0 (dB) at each dwell as a unit recedes out of range.

    Distance grows linearly from 1 to d_far units (constant radial velocity);
    path loss uses the log-distance model L = 10*exponent*log10(d/d0); a small
    dwell-to-dwell Gaussian wobble stands in for shadowing/scintillation.
    Seeded, so every comparison run sees the same trace. The trace covers all
    four thresholds of the threshold table (7.0 .. 22.8 dB Es/N0), so the
    adaptive chain sails across every switching boundary on its way down.
    """
    rng = np.random.default_rng(seed)
    d = np.linspace(1.0, d_far, n_dwells)
    loss = 10.0 * exponent * np.log10(d)
    wobble = rng.normal(0.0, wobble_db, n_dwells)
    return snr0_db - loss + wobble


def run_adaptive_vs_fixed(target_ber=1e-3, symbol_rate=1e6, n_dwells=40, seed=7):
    """Adaptive link vs four fixed schemes across a receding-unit SNR trace.

    Run with: python main.py --adaptive-vs-fixed

    One synthetic SNR trace (synthetic_snr_trace) is offered to five full
    chains - the four fixed modulations (BPSK/QPSK/16-QAM/64-QAM) and an
    adaptive chain steered by adapt.py's threshold table with its
    1 dB hysteresis guard. Every dwell's BER is re-measured with the same
    symbol-level Monte-Carlo chain (monte_carlo_ber, seed 42) that produced
    the thresholds, so verdicts rest on the simulator, not on closed forms.
    Per-dwell throughput = bits/symbol * symbol_rate, set to zero for any dwell
    whose measured BER violates the target (the link is off-air there).

    Two stacked panels: BER-vs-SNR (log) and throughput-vs-SNR for the
    adaptive chain overlaid on all four fixed schemes, with the adaptive
    switching points marked on the throughput plot. Saves
    results/adaptive_vs_fixed.png. Closed-form BER curves are dotted onto the
    top panel purely as a smooth eye-guide for where each fixed scheme's
    crossing sits.
    """
    from adapt import AdaptiveModulation, link_availability  # no main imports

    snr = synthetic_snr_trace(n_dwells=n_dwells, seed=seed)
    selector = AdaptiveModulation()               # threshold table + 1 dB hysteresis
    modes = [selector.update(e) for e in snr]
    n_dwells = np.asarray(snr).size
    switch_idx = [i for i in range(1, n_dwells) if modes[i] != modes[i - 1]]

    # One Monte-Carlo run per (scheme, Eb/N0) point, shared by every chain
    # that needs it, so the fixed curves and the adaptive curve agree exactly.
    ber_memo = {}

    def ber_at(es_db, scheme):
        eb_db = float(es_db - 10.0 * np.log10(BITS_PER_SYMBOL[scheme]))
        key = (scheme, round(eb_db, 6))
        if key not in ber_memo:
            ber_memo[key] = monte_carlo_ber(scheme, eb_db, seed=42)
        return ber_memo[key]

    fixed_ber = {s: np.array([ber_at(e, s) for e in snr]) for s in ALL_SCHEMES}
    adapt_ber = np.array([ber_at(e, m) for e, m in zip(snr, modes)])

    def throughput(k_arr, ber_arr):
        return np.where(ber_arr < target_ber, np.asarray(k_arr) * symbol_rate,
                        0.0)

    fixed_thr = {s: throughput(np.full(n_dwells, BITS_PER_SYMBOL[s]),
                               fixed_ber[s]) for s in ALL_SCHEMES}
    adapt_k = np.array([BITS_PER_SYMBOL[m] for m in modes])
    adapt_thr = throughput(adapt_k, adapt_ber)

    print(f"\n=== Adaptive vs fixed across a receding-unit trace "
          f"({n_dwells} dwells, Rs = {symbol_rate:.0e} symb/s, "
          f"target BER < {target_ber:.0e}) ===")
    print("  dwell: Es/N0 stays in range, noise-scaled replay of the same "
          "Monte-Carlo chain as the threshold table (seed 42).")
    for s in ALL_SCHEMES:
        on_air = int(np.count_nonzero(fixed_thr[s]))
        mean_mbps = float(np.mean(fixed_thr[s]) / 1e6)
        print(f"  fixed {s:>6}: {on_air:>3}/{n_dwells} dwells on-air, "
              f"mean throughput {mean_mbps:5.2f} Mbit/s")
    on_air = int(np.count_nonzero(adapt_thr))
    mean_mbps = float(np.mean(adapt_thr) / 1e6)
    print(f"  adaptive : {on_air:>3}/{n_dwells} dwells on-air, "
          f"mean throughput {mean_mbps:5.2f} Mbit/s  "
          f"(cumulative bits: {int(adapt_thr.sum()):,})")
    print("  mode timeline (dwell counts): ", end="")
    counts = {s: sum(1 for m in modes if m == s) for s in ALL_SCHEMES}
    for s in ALL_SCHEMES:
        if counts[s]:
            print(f"{s} x{counts[s]}", end="  ")
    print()
    print("  switching points (dwell, Es/N0 dB, -> mode):")
    for i in switch_idx:
        print(f"    dwell {i:>3} @ {snr[i]:5.2f} dB  -> {modes[i]}")

    # --- link availability over the trace --------------------------------
    availability = {"adaptive": link_availability(adapt_ber, target_ber)}
    availability.update({s: link_availability(fixed_ber[s], target_ber)
                         for s in ALL_SCHEMES})
    on_air = {**{s: int(np.count_nonzero(fixed_thr[s])) for s in ALL_SCHEMES},
              "adaptive": int(np.count_nonzero(adapt_thr))}
    print("\n  link availability (% of trace with BER < "
          f"{target_ber:.0e}):")
    for s in ["adaptive", *ALL_SCHEMES]:
        print(f"    {s:>8}: {availability[s]:5.1f}%  "
              f"({on_air[s]}/{n_dwells} dwells on-air)")

    import csv
    import os
    os.makedirs("results", exist_ok=True)
    csv_path = "results/availability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "availability_pct", "on_air_dwells",
                         "total_dwells", "mean_throughput_mbps"])
        for s in ["adaptive", *ALL_SCHEMES]:
            mean_mbps = (float(np.mean(adapt_thr) / 1e6) if s == "adaptive"
                         else float(np.mean(fixed_thr[s]) / 1e6))
            writer.writerow([s, f"{availability[s]:.1f}", on_air[s],
                             n_dwells, f"{mean_mbps:.2f}"])
    print(f"  availability table saved to {csv_path}")

    # --- plot ---------------------------------------------------------------
    fig, (ax, ax_thr) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    eb_snr = np.arange(0.0, 30.0, 0.01)
    for s in ALL_SCHEMES:
        color, marker = COLORS[s], MARKERS[s]
        ok = fixed_ber[s] > 0
        ax.semilogy(snr[ok], fixed_ber[s][ok], color=color, marker=marker,
                    markersize=3.5, linewidth=1.0, alpha=0.85, label=f"fixed {s}")
        ax.semilogy(eb_snr, theoretical_ber(s, eb_snr), color=color,
                    linestyle=":", linewidth=1.0, alpha=0.5)
    ax.semilogy([snr.min(), snr.max()], [target_ber, target_ber], color="k",
                linestyle="--", linewidth=1.0, label=f"BER target {target_ber:.0e}")

    # adaptive runs are drawn per mode, thick line inheriting the mode colour.
    run0, mode = 0, modes[0]
    for i in range(1, n_dwells + 1):
        if i == n_dwells or modes[i] != mode:
            seg = slice(run0, i)
            ax.semilogy(snr[seg], adapt_ber[seg], color=COLORS[mode],
                        linewidth=2.6, zorder=4)
            run0, mode = i, modes[i] if i < n_dwells else mode
    ax.plot([], [], color="k", linewidth=2.6,
            label="adaptive (thick run, mode = colour)")
    ax.set_ylabel("per-dwell BER")
    ax.set_title("Adaptive link vs fixed schemes, unit receding out of range "
                 "(dotted = closed-form eye-guide)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    ymax_mbps = max(BITS_PER_SYMBOL.values()) * symbol_rate / 1e6
    for s in ALL_SCHEMES:
        ax_thr.plot(snr, fixed_thr[s] / 1e6, color=COLORS[s], linewidth=1.1,
                    alpha=0.75, label=f"fixed {s}")
    run0, mode = 0, modes[0]
    for i in range(1, n_dwells + 1):
        if i == n_dwells or modes[i] != mode:
            seg = slice(run0, i)
            ax_thr.plot(snr[seg], adapt_thr[seg] / 1e6, color=COLORS[mode],
                        linewidth=2.6, zorder=4)
            run0, mode = i, modes[i] if i < n_dwells else mode
    ax_thr.plot([], [], color="k", linewidth=2.6,
                label="adaptive (thick run, mode = colour)")
    ax_thr.scatter(snr[switch_idx], adapt_thr[switch_idx] / 1e6, color="k",
                   marker="o", s=45, zorder=5, label="mode switch")
    for i in switch_idx:
        ax_thr.axvline(snr[i], color="gray", linestyle=":", linewidth=1.0)
        ax_thr.annotate(f"{modes[i]}", xy=(snr[i], ymax_mbps), fontsize=9,
                        color=COLORS[modes[i]], ha="center", va="top",
                        annotation_clip=False)
    ax_thr.set_ylim(0, ymax_mbps * 1.18)
    ax_thr.set_ylabel("per-dwell throughput (Mbit/s)")
    ax_thr.set_xlabel(r"$E_s/N_0$ (dB)")
    ax_thr.set_title(r"Per-dwell throughput $= k \cdot R_s$ when BER below "
                     r"target, else 0")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out = "results/adaptive_vs_fixed.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive-modulation link simulation (BER sweeps).")
    parser.add_argument(
        "--fading", action="store_true",
        help="re-run the symbol-level and pulse-shaped BER sweeps with flat "
             "Rayleigh fading enabled vs the AWGN-only curve, per scheme; "
             "saves results/fading_vs_awgn.png")
    parser.add_argument(
        "--adaptive-vs-fixed", action="store_true",
        help="compare the adaptive link against the four fixed schemes across a "
             "receding-unit SNR trace (BER + per-dwell throughput); saves "
             "results/adaptive_vs_fixed.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_bpsk_check()
    results = run_all_schemes_check()
    verify_shaped_regression(results)
    if args.fading:
        run_fading_comparison(results)
    if args.adaptive_vs_fixed:
        run_adaptive_vs_fixed()