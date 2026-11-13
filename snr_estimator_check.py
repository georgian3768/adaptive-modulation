# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Synthetic time-varying SNR trace through the pilot SNR estimator.

Checks how accurately receiver.estimate_snr tracks a channel whose true Es/N0
drifts over time, and shows what adapt.select_modulation would do with the
estimate.

Method: a fresh QPSK pilot block is drawn per slot (fixed seed) and sent over
AWGN at the slot's true Es/N0 via channel.add_awgn; the receiver estimates
Es/N0 from the received block. Estimate vs true are printed side by side with
the absolute error, plus the scheme the selector picks. The selector's
threshold table is built from the SAME closed-form BER theory the BER sweeps
validate, at a 1e-3 target (converted to Es/N0 so the units match the
estimator). Also saves results/snr_estimator_check.png.

Run:  .venv/bin/python snr_estimator_check.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modem import modulate, BITS_PER_SYMBOL
from channel import add_awgn
from receiver import estimate_snr, SNR_DB_FLOOR, SNR_DB_CEIL
from adapt import select_modulation
from main import theoretical_ber

ALL_SCHEMES = ["bpsk", "qpsk", "16qam", "64qam"]
TARGET_BER = 1e-3
N_PILOTS = 256           # pilot symbols per frame - tunable accuracy knob
N_SLOTS = 45             # length of the synthetic trace
SEED = 7


def _snr_for_target_ber(scheme, eb_grid, target=TARGET_BER):
    """Es/N0 (dB) where the scheme's closed-form BER crosses `target`.

    Inverts theoretical_ber (an Eb/N0 quantity) and converts the crossing to
    Es/N0 dB = Eb/N0 dB + 10*log10(k) so the units match the estimator.
    """
    ber = theoretical_ber(scheme, eb_grid)
    hit = int(np.argmax(ber < target))
    if ber[hit] >= target:
        return None
    lo = hit - 1
    t = np.log(target / ber[lo]) / np.log(ber[hit] / ber[lo])
    eb_db = eb_grid[lo] + t * (eb_grid[hit] - eb_grid[lo])
    return eb_db + 10.0 * np.log10(BITS_PER_SYMBOL[scheme])


def build_threshold_table():
    """{scheme: es_n0_db_min} from theory at TARGET_BER, ascending at use-site.

    The most robust scheme doubles as the implicit fallback (its threshold is
    simply the lowest), so no -inf key is needed.
    """
    eb_grid = np.arange(-5.0, 40.0, 0.05)
    return {s: _snr_for_target_ber(s, eb_grid) for s in ALL_SCHEMES}


def main():
    table = build_threshold_table()
    print(f"Threshold table (Es/N0 dB required for BER <= {TARGET_BER:.0e}):")
    for scheme, min_db in sorted(table.items(), key=lambda kv: kv[1]):
        print(f"   {scheme:6s}: >= {min_db:5.2f} dB")
    print(f"   {'bpsk':6s}: always usable (fallback)")

    rng = np.random.default_rng(SEED)
    k = BITS_PER_SYMBOL["qpsk"]
    pilots = modulate(rng.integers(0, 2, N_PILOTS * k), "qpsk")

    rows = []
    for i in range(N_SLOTS):
        # Smoothly drifting channel: ramp 2..28 dB plus a gentle wobble.
        frac = i / (N_SLOTS - 1)
        true_db = 2.0 + 26.0 * frac + 1.2 * np.sin(2.0 * np.pi * frac)

        rx, _ = add_awgn(pilots, true_db - 10.0 * np.log10(k),
                         bits_per_symbol=k, rng=rng)
        est_db = estimate_snr(rx, pilots)
        scheme_est = select_modulation(est_db, table)
        scheme_true = select_modulation(true_db, table)
        rows.append((i, true_db, est_db, est_db - true_db,
                     scheme_est, scheme_true))

    print(f"\n{'slot':>4}  {'true Es/N0':>10}  {'est Es/N0':>10}  "
          f"{'err dB':>7}  {'scheme':>6}   agree")
    for i, t, e, err, s_est, s_true in rows:
        agree = "ok" if s_est == s_true else "X"
        print(f"{i:>4}  {t:>10.2f}  {e:>10.2f}  {err:>+7.2f}  {s_est:>6}  {agree}")

    errs = np.array([r[3] for r in rows])
    n_mismatch = sum(1 for r in rows if r[4] != r[5])
    print(f"\nEstimator accuracy over {len(rows)} slots:")
    print(f"    bias          = {errs.mean():+.3f} dB")
    print(f"    RMS error     = {np.sqrt(np.mean(errs ** 2)):.3f} dB")
    print(f"    max |error|   = {np.abs(errs).max():.3f} dB "
          f"(at slot {int(np.argmax(np.abs(errs)))})")
    print(f"    modulation picks matching truth: {len(rows) - n_mismatch}/{len(rows)}")

    # --- plot: the trace and the estimate-vs-truth scatter ------------------
    times = np.arange(N_SLOTS)
    true_arr = np.array([r[1] for r in rows])
    est_arr = np.array([r[2] for r in rows])
    picked = np.array([ALL_SCHEMES.index(r[4]) for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(times, true_arr, "b-", linewidth=1.5, label="true Es/N0")
    ax.plot(times, est_arr, "o", color="C3", markersize=4,
            label=f"est Es/N0 (pilot block, N={N_PILOTS})")
    for scheme, db in sorted(table.items(), key=lambda kv: kv[1]):
        if not np.isfinite(db):
            continue
        ax.axhline(db, linestyle=":", color="gray", linewidth=0.7)
    ax.set_ylabel("Es/N0 (dB)")
    ax.set_title("Pilot-based SNR estimator on a synthetic time-varying channel")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax1 = axes[1]
    scheme_colors = [plt.get_cmap("tab10")(j) for j in range(4)]
    ax1.plot(true_arr, true_arr, "k--", linewidth=1.0,
             label="ideal (est = true)")
    ax1.scatter(true_arr, est_arr,
                c=[scheme_colors[p] for p in picked], s=28, zorder=3)
    for j, scheme in enumerate(ALL_SCHEMES):
        ax1.scatter([], [], c=[scheme_colors[j]], s=28, label=f"select -> {scheme}")
    ax1.set_xlabel("true Es/N0 (dB)")
    ax1.set_ylabel("estimated Es/N0 (dB)")
    ax1.set_xlim(SNR_DB_FLOOR, SNR_DB_CEIL)
    ax1.set_ylim(SNR_DB_FLOOR, SNR_DB_CEIL)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("Estimator vs truth")

    plt.tight_layout()
    out = "results/snr_estimator_check.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()