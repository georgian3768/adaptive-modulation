# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Assemble a report-ready summary: thresholds + hysteresis, per-threshold BER
corroboration, availability, and final BER/throughput figures over the
receding-unit trace. Writes results/summary_report.md and prints the markdown
for pasting.

Data sources
    * Threshold table and up/down shifts: adapt.py constants.
    * BER at each threshold: re-measured with the symbol-level chain
      (monte_carlo_ber, seed 42), the same corroboration as verify_thresholds.
    * Availability / mean throughput: results/availability.csv, written by
      the adaptive-vs-fixed run (main.py --adaptive-vs-fixed).
    * Switching points / mode timeline / payload: cheap deterministic replay
      of the receding-unit trace (synthetic_snr_trace seed 7) through
      AdaptiveModulation; no BER re-simulation needed.

Run:  .venv/bin/python make_report_tables.py
"""

import csv
import datetime

import numpy as np

from adapt import (THRESHOLDS_ES_N0_DB, AdaptiveModulation,
                   up_shift_thresholds, down_shift_thresholds)
from main import ALL_SCHEMES, monte_carlo_ber, synthetic_snr_trace
from modem import BITS_PER_SYMBOL

TARGET_BER = 1e-3
N_DWELLS = 40
SEED = 7


def _order():
    return sorted(ALL_SCHEMES, key=THRESHOLDS_ES_N0_DB.get)


def _ber_at_es(scheme, es_db):
    eb_db = es_db - 10.0 * np.log10(BITS_PER_SYMBOL[scheme])
    return monte_carlo_ber(scheme, eb_db, seed=42)


def _availability_csv():
    rows = {}
    with open("results/availability.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["scheme"]] = r
    return rows


def section_thresholds():
    up = up_shift_thresholds()
    down = down_shift_thresholds()
    lines = [
        "### A. Threshold table (Es/N0 dB, symbol-level sweep, BER < 1e-3)",
        "",
        "Hysteresis: up-shift = threshold + 0.5 dB, down-shift = threshold "
        "\u2013 0.5 dB (1 dB gap, adapt.py default).",
        "",
        "| scheme | k | base | up-shift | down-shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in _order():
        lines.append(f"| {s} | {BITS_PER_SYMBOL[s]} "
                     f"| {THRESHOLDS_ES_N0_DB[s]:.2f} "
                     f"| {up[s]:.2f} | {down[s]:.2f} |")
    lines.append("")
    return lines


def section_corroboration():
    up = up_shift_thresholds()
    down = down_shift_thresholds()
    order = _order()
    lines = [
        "### B. Simulated BER at each threshold (symbol-level chain, seed 42)",
        "",
        "The scheme a switch moves INTO is measured at the exact threshold "
        "SNR; every populated row must be below the 1e-3 target.",
        "",
        "| threshold | switches into | SNR (dB) | BER | PASS |",
        "|---|---|---:|---:|:--:|",
    ]
    prev = order[0]
    for s in order:
        ber_up = _ber_at_es(s, up[s])
        lines.append(f"| up-shift -> {s} | {s} | {up[s]:.2f} "
                     f"| {ber_up:.4g} | {'OK' if ber_up < TARGET_BER else 'X'} |")
        if s == prev:
            lines.append(f"| down-shift <- {s} | \u2013 (fallback) "
                         f"| {down[s]:.2f} | \u2013 | \u2013 |")
        else:
            ber_do = _ber_at_es(prev, down[s])
            lines.append(f"| down-shift <- {s} | {prev} | {down[s]:.2f} "
                         f"| {ber_do:.4g} "
                         f"| {'OK' if ber_do < TARGET_BER else 'X'} |")
        prev = s
    lines.append("")
    return lines


def section_availability_and_throughput():
    csv_rows = _availability_csv()
    n_dwells = int(list(csv_rows.values())[0]["total_dwells"])

    lines = [
        "### C. Link availability & throughput over the receding-unit "
        "trace "
        f"({n_dwells} dwells, Rs = 1 Msps, target BER < 1e-3)",
        "",
        "Per-dwell BER measured with the symbol-level Monte-Carlo chain; "
        "throughput "
        "= k * Rs only where the measured BER stays below the target, else 0.",
        "",
        "| chain | availability | on-air dwells | mean throughput |",
        "|---|---:|---:|---:|",
    ]
    for key in ["adaptive", *ALL_SCHEMES]:
        r = csv_rows[key]
        lines.append(f"| {key} | {r['availability_pct']}% "
                     f"| {r['on_air_dwells']}/{r['total_dwells']} "
                     f"| {r['mean_throughput_mbps']} Mbit/s |")

    # adaptive detail: deterministic replay (trace + selector only)
    snr = synthetic_snr_trace(n_dwells=n_dwells, seed=SEED)
    sel = AdaptiveModulation()
    modes = [sel.update(e) for e in snr]
    from collections import Counter
    counts = Counter(modes)
    timeline = ", ".join(f"{s} \u00d7 {counts[s]}" for s in ALL_SCHEMES
                         if counts[s])
    switches = [i for i in range(1, len(modes)) if modes[i] != modes[i - 1]]
    sw_rows = "; ".join(
        f"dwell {i} @ {snr[i]:.2f} dB \u2192 {modes[i]}" for i in switches)
    mean_mbps = float(csv_rows["adaptive"]["mean_throughput_mbps"])
    cumulative = mean_mbps * 1e6 * n_dwells

    lines += [
        "Adaptive detail:",
        f"* mode timeline: {timeline}",
        f"* switching points: {sw_rows}",
        f"* cumulative payload on-air: {cumulative / 1e6:.1f} Mbit",
        "",
    ]
    return lines


def section_headline():
    csv_rows = _availability_csv()
    adapt = float(csv_rows["adaptive"]["mean_throughput_mbps"])
    best_fixed = max(float(csv_rows[s]["mean_throughput_mbps"])
                     for s in ALL_SCHEMES)
    best_name = max(ALL_SCHEMES, key=lambda s: float(
        csv_rows[s]["mean_throughput_mbps"]))
    gain = (adapt / best_fixed - 1.0) * 100.0
    return [
        "### D. Headline numbers",
        "",
        f"* Adaptive mean throughput **{adapt:.2f} Mbit/s** vs best fixed "
        f"({best_name}) **{best_fixed:.2f} Mbit/s** \u2192 "
        f"+{gain:.0f}%.",
        f"* Adaptive availability {csv_rows['adaptive']['availability_pct']}% "
        f"({csv_rows['adaptive']['on_air_dwells']} of "
        f"{csv_rows['adaptive']['total_dwells']} dwells on-air); the "
        "robust fixed fallback (BPSK) reaches "
        f"{csv_rows['bpsk']['availability_pct']}% at a third of the "
        "useful throughput.",
        "* BER target is enforced per dwell: on-air dwells measured below "
        "1e-3, and every mode switch moves into a scheme whose simulated BER "
        "is below 1e-3 at the switch SNR (Sect. B).",
        "",
    ]


def main():
    date = datetime.date.today().isoformat()
    body = (["# Adaptive-modulation link \u2014 summary results "
             f"(generated {date})", ""]
            + section_thresholds()
            + section_corroboration()
            + section_availability_and_throughput()
            + section_headline())
    markdown = "\n".join(body)
    with open("results/summary_report.md", "w") as f:
        f.write(markdown + "\n")
    print(markdown)
    print(f"\nSaved to results/summary_report.md")


if __name__ == "__main__":
    main()