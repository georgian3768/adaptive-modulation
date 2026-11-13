# Adaptive-modulation link — summary results (generated 2026-09-28)

### A. Threshold table (Es/N0 dB, symbol-level sweep, BER < 1e-3)

Hysteresis: up-shift = threshold + 0.5 dB, down-shift = threshold – 0.5 dB (1 dB gap, adapt.py default).

| scheme | k | base | up-shift | down-shift |
|---|---:|---:|---:|---:|
| bpsk | 1 | 7.00 | 7.50 | 6.50 |
| qpsk | 2 | 10.01 | 10.51 | 9.51 |
| 16qam | 4 | 17.02 | 17.52 | 16.52 |
| 64qam | 6 | 22.78 | 23.28 | 22.28 |

### B. Simulated BER at each threshold (symbol-level chain, seed 42)

The scheme a switch moves INTO is measured at the exact threshold SNR; every populated row must be below the 1e-3 target.

| threshold | switches into | SNR (dB) | BER | PASS |
|---|---|---:|---:|:--:|
| up-shift -> bpsk | bpsk | 7.50 | 0.0004708 | OK |
| down-shift <- bpsk | – (fallback) | 6.50 | – | – |
| up-shift -> qpsk | qpsk | 10.51 | 0.0004583 | OK |
| down-shift <- qpsk | bpsk | 9.51 | 1.019e-05 | OK |
| up-shift -> 16qam | 16qam | 17.52 | 0.0003083 | OK |
| down-shift <- 16qam | qpsk | 16.52 | 0 | OK |
| up-shift -> 64qam | 64qam | 23.28 | 0.0004042 | OK |
| down-shift <- 64qam | 16qam | 22.28 | 0 | OK |

### C. Link availability & throughput over the receding-unit trace (40 dwells, Rs = 1 Msps, target BER < 1e-3)

Per-dwell BER measured with the symbol-level Monte-Carlo chain; throughput = k * Rs only where the measured BER stays below the target, else 0.

| chain | availability | on-air dwells | mean throughput |
|---|---:|---:|---:|
| adaptive | 42.5% | 17/40 | 1.18 Mbit/s |
| bpsk | 47.5% | 19/40 | 0.47 Mbit/s |
| qpsk | 35.0% | 14/40 | 0.70 Mbit/s |
| 16qam | 15.0% | 6/40 | 0.60 Mbit/s |
| 64qam | 5.0% | 2/40 | 0.30 Mbit/s |
Adaptive detail:
* mode timeline: bpsk × 24, qpsk × 10, 16qam × 4, 64qam × 2
* switching points: dwell 2 @ 21.72 dB → 16qam; dwell 6 @ 16.05 dB → qpsk; dwell 16 @ 8.23 dB → bpsk
* cumulative payload on-air: 47.2 Mbit

### D. Headline numbers

* Adaptive mean throughput **1.18 Mbit/s** vs best fixed (qpsk) **0.70 Mbit/s** → +69%.
* Adaptive availability 42.5% (17 of 40 dwells on-air); the robust fixed fallback (BPSK) reaches 47.5% at a third of the useful throughput.
* BER target is enforced per dwell: on-air dwells measured below 1e-3, and every mode switch moves into a scheme whose simulated BER is below 1e-3 at the switch SNR (Sect. B).

