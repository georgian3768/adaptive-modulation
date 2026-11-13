# Adaptive-Modulation Live Demo — Presenter Script & Reference

Scope: a **10–15 minute** technical presentation of the adaptive-modulation link
demo in this workspace. Part 1 is a timed narrator script (what to say, when to
press which key, which figure to show). Part 2 is the reference manual: run
commands, key numbers, screen tour, figure guide, glossary, and a Q&A bank.

Every number quoted here is taken directly from the outputs in this repo
(`results/`, `verify_thresholds.py`, `demo.py`). Nothing needs to be memorized beyond
the handful of "money numbers" flagged with ⭐.

---

## Part 1 — Narrator script (~13 min)

### 0. Hook (30 s)

> "Every wireless link tomorrow wants to be as fast as possible for as long as
> possible. The problem: a radio that is close and loud can use a fast, dense
> modulation like 64-QAM — but as the transmitter recedes and the signal
> fades, that same choice becomes useless. This demo watches a link make that
> choice **adaptively**, on the fly, and shows it beating every fixed choice."

### 1. The architecture (2 min) — show `results/link_block_diagram.png` (or .svg)

> "The chain is a textbook transmitter-to-receiver path." Point as you speak:
>
> - **Modulator** — picks one of four modulations `{BPSK, QPSK, 16-QAM,
>   64-QAM}`. All are **unit-energy** constellations and Gray-coded.
> - **RRC pulse shaping** — a square-root-raised-cosine filter
>   (β = 0.22). Together with the matched filter on the receive side it is
>   **ISI-free at the sampling instants** — that is the reason we can swap
>   modulation without touching synchronization.
> - **Channel** — AWGN plus flat Rayleigh fading. Not shown live today, but
>   verified separately.
> - **RRC matched filter + sampler** — recovers soft symbols.
> - **SNR estimator** — measures **Es/N0 per dwell** using a known pilot
>   block (~0.3 dB RMS accuracy).
> - **Demodulator** — nearest-point decisions on the fixed lattice.
> - **The feedback arrow** — the star of the show. The SNR estimate drives a
>   threshold table with hysteresis; the selected mode is fed back to the
>   modulator for the **next dwell**."

> (Say:) "Only the estimator feeds back — demodulation stays a fixed
> nearest-point decision. That keeps the adaptive part pure and provable."

### 2. Why the decision is a *threshold table* (3 min) — show
`results/ber_all_schemes.png` and quote the threshold table

> "The modulator decision is data, gathered once in a symbol-level sweep: a
> Monte-Carlo BER
> sweep of every scheme versus SNR. The rule: a scheme is usable when its **BER
> stays below 1e-3**. Reading where each curve crosses gives the threshold
> table — minimum Es/N0 in dB:"

| scheme | k (bits/sym) | min Es/N0 for BER < 1e-3 |
|---|---:|---:|
| BPSK | 1 | 7.00 dB |
| QPSK | 2 | 10.01 dB |
| 16-QAM | 4 | 17.02 dB |
| 64-QAM | 6 | 22.78 dB |

> "See the pattern? **Each doubling of bits-per-symbol costs roughly 3–7 dB of
> signal.** 64-QAM is 15 dB more demanding than BPSK. If the link is close
> enough to afford it, 64-QAM moves 6× the bits of BPSK in the same symbol —
> that is the 6× throughput upside. Two notes on units:"
>
> - "Everything here is **Es/N0** (energy per *symbol*), because that is what
>   our SNR estimator measures. The liter ature's Eb/N0 (per *bit*) relates by
>   Es/N0 = Eb/N0 + 10·log10(k). The thresholds you see in `summary_report.md`
>   are listed in Es/N0."
> - "⏱ every threshold is **verified against simulation**: at each switch SNR,
>   the scheme you move *into* is measured with the full Monte-Carlo chain and
>   comes out below 1e-3. That check is automated in `verify_thresholds.py`."

### 3. Hysteresis — why the naive version chatters (2 min)

> "A plain threshold lookup has a nasty failure. If the SNR wobbles up and
> down across a boundary — say ±0.3 dB around 17 dB — a stateless selector
> flips between QPSK and 16-QAM **every other dwell**. Re-tuning and
> re-training the link constantly is expensive and looks broken on a demo."
>
> "Fix: two thresholds per boundary instead of one."
> - ⭐ **Up-shift** = threshold **+ 0.5 dB** — upgrade only when clearly above.
> - ⭐ **Down-shift** = threshold **− 0.5 dB** — keep the mode until clearly
>   below.
>
> "The 1 dB dead band means: once you are in 16-QAM you stay there until the
> SNR *really* drops. Verified: an SNR oscillating ±0.5 dB around a boundary
> for 100 calls produces exactly **one** switch, never dozens."

### 4. The scenario & the metrics (1 min)

> "Now the story: a unit **moving out of range**. Simulated trace: constant
> radial velocity, log-distance path loss (`SNR ≈ 27 dB → 1 dB`), plus small
> dwelling-to-dwelling wobble to make it honest. 40 dwells; each dwell is one
> observation window."
>
> "Per dwell we measure: **BER** (via the same validated simulation), and
> **throughput** = k × symbol-rate (1 Msps), but **zero whenever the BER
> target is violated** — the link is off-air. We also track **availability**:
> percentage of dwells that are on-air."

### 5. THE LIVE DEMO (4 min) — `python demo.py`

> "Let me run it. Watch two panels."

```
.venv/bin/python demo.py --rt 0.8
```

> **Narration while it runs** (two panels):
>
> - **Left = constellation.** The dark points are the current mode's codebook;
>   the scattered cloud is the received symbols including noise. Watch the
>   cloud *shrink* as SNR is high, then *spread* as the link recedes.
>   ⭐ The constellation **changes live** when we switch modes — that is the
>   moment to point at the screen.
> - **Right = SNR trace.** The thin dotted lines are the four decision
>   thresholds; the black dot is "you are here". A red **"SWITCH → 16-QAM"**
>   flashes on both panels at every mode change.
>
> What the terminal prints (one line per dwell):
>
> ```
> dwell  Es/N0     mode    BER      link  thr[kb]  running cum(Mbit)
>    1  27.00    64qam  3.0e-07   OK     6.00         6.00
>    3  21.72    16qam  1.9e-08   OK     4.00        16.00
> ```
>
> **Expected timeline — narrate it as it happens** (⌘ dwells are 1-indexed):
>
> 1. **Start, after you press Enter** — SNR ≈ 27 dB: the link boots into
>    the greedy upgrade path straight to **64-QAM** (dwells 1–2), 6 Mbit/s.
> 2. **Dwell 3 @ 21.72 dB** — below 64-QAM's down-shift (22.28): drop to
>    **16-QAM**. "It downgrades only when it must — hysteresis."
> 3. **Dwell 7 @ 16.05 dB** — below 16-QAM's down-shift (16.52): drop to
>    **QPSK**.
> 4. **Dwell 17 @ 8.23 dB** — below QPSK's down-shift (9.51): drop to
>    **BPSK**, the most robust mode.
> 5. **The end of the trace** — SNR ≈ 1 dB: even BPSK's BER estimate crosses
>    the target; throughput goes to 0, the link is out of range. Exactly what
>    "moving out of range" should look like.
>
> **If asked about speed:** "This demo uses the closed-form BER (validated
> against the sim) purely so it updates instantly; the authoritative report
> run uses the full Monte-Carlo chain — the reported numbers come from that."

### 6. Results recap (1.5 min) — show `results/adaptive_vs_fixed.png`

> "Now the payoff — same trace, five links: the four fixed modulations and the
> adaptive one." Point to the table (also in `availability.csv`):

| link | availability | mean throughput |
|---|---:|---:|
| fixed BPSK | 47.5% | 0.47 Mbit/s |
| fixed QPSK | 35.0% | 0.70 Mbit/s |
| fixed 16-QAM | 15.0% | 0.60 Mbit/s |
| fixed 64-QAM | 5.0% | 0.30 Mbit/s |
| ⭐ **adaptive** | **42.5%** | **1.18 Mbit/s** |

> ⭐ "The adaptive link is **+69%** over the best fixed scheme (QPSK's 0.70),
> while keeping BPSK-like robustness (42.5% availability vs BPSK's 47.5% —
> a tiny 5-point availability cost, for more than double the data). That is
> the whole pitch in one sentence: **the adaptive link captures most of the
> robustness of the slow mode and most of the speed of the fast modes.**"

### 7. Close + Q&A cues (1 min)

> "Open questions — none blocking, but the natural next steps: learn the
> thresholds from live estimates instead of a fixed table; add closed-loop
> feedback with estimation error; and test real fading plus imperfect
> estimation together. Happy to dig into any of it."

Expected questions & answers are in Part 2.

---

## Part 2 — Reference manual

### A. Run commands (from the repo root)

| What | Command | Notes |
|---|---|---|
| Live demo (needs display) | `.venv/bin/python demo.py` | 40 dwells, 0.35 s/dwell |
| Slow enough to narrate | `.venv/bin/python demo.py --rt 0.8` | ≈ 32 s of runtime |
| Headless / CI (print only) | `.venv/bin/python demo.py --fast` | no window, no sleep |
| Headless + final picture | `.venv/bin/python demo.py --fast --save` | writes `results/demo_constellation_final.png` |
| Interrupt mid-demo | Ctrl-C | prints the running summary |
| Full Monte-Carlo receding-unit run | `.venv/bin/python main.py --adaptive-vs-fixed` | slow (~1–2 min); regenerates `adaptive_vs_fixed.png` + `availability.csv` |
| Threshold gate check | `.venv/bin/python verify_thresholds.py` | re-verifies table + BER at thresholds + no-chatter |
| Block diagram | `.venv/bin/python block_diagram.py` | `.svg` + `.png` |
| Report tables | `.venv/bin/python make_report_tables.py` | ~1 min; writes `results/summary_report.md` |
| Interactive GUI (needs display) | `.venv/bin/python interactive_demo.py` | channel sliders/toggles, Step/Auto/Reset, live TX/RX panels |
| Interactive GUI, print only | `.venv/bin/python interactive_demo.py --headless 40` | distance 1→15 ramp, dwell table |
| Tests | `.venv/bin/python -m unittest discover -s tests` | 48 tests |

### B. Money numbers (memorize these) ⭐

- Thresholds (Es/N0, BER < 1e-3): BPSK **7.00**, QPSK **10.01**, 16-QAM **17.02**,
  64-QAM **22.78** dB.
- Hysteresis: up-shift = threshold **+0.5 dB**, down-shift = **−0.5 dB** (1 dB gap);
  greedy up, one-level down.
- Scenario: unit recedes, Es/N0 ≈ **27 → 1 dB** over **40 dwells**; Rs = 1 Msps.
- Per-dwell throughput: **k × Rs, or 0 if measured BER ≥ 1e-3**.
- ⭐ Adaptive **1.18 Mbit/s** vs best fixed (QPSK) **0.70 Mbit/s** → **+69%**;
  availability adaptive **42.5%** vs BPSK **47.5%**.
- Estimator quality: bias +0.07 dB, **RMS 0.33 dB**, max err 0.67 dB.
- Demo (closed-form): 19/40 on-air, 50 Mbit cumulative; report (Monte-Carlo):
  17/40 on-air, 47.2 Mbit. → the small gap is boundary measurement noise.
- Mode timeline of the trace: `64-QAM ×2 → 16-QAM ×4 → QPSK ×10 → BPSK ×24`,
  switching at 21.72 / 16.05 / 8.23 dB Es/N0.

### C. What you will see on screen (demo)

- **Window**: one figure, two panels.
- **Left — constellation**: dark dots = codebook of the active mode; colored
  cloud = noisy received samples for the current dwell. Title shows mode,
  Es/N0, BER estimate. Panel covers the 64-QAM extent (±3 on both axes), so
  lower modes "shrink" visually.
- **Right — SNR trace**: gray trace (history), dotted lines = the four up-shift
  thresholds (labeled), black dot = current dwell. Updates each `--rt`.
- **On a switch**: red `SWITCH → <mode>` flashes on both panels; the mode
  label in the constellation title changes.
- **Terminal**: one compact row per dwell: `dwell | Es/N0 | mode | BER | link
  OK/OUT | current k·Rs (Mbit/s) | running cumulative (Mbit)`.
- **At the end**: summary block — on-air fraction (availability %), cumulative
  payload in Mbit, last mode.

### D. Figures in `results/` — what to show when

| File | Shows | Use for |
|---|---|---|
| `link_block_diagram.png` / `.svg` | full chain + feedback | Architecture slide (Part 1 §1) |
| `ber_all_schemes.png` | sim vs theory, 4 schemes | Threshold rationale (§2) |
| `adaptive_vs_fixed.png` | BER-vs-SNR + throughput-vs-SNR, 5 chains, switch markers | Results (§6) |
| `availability.csv` / table in report | the comparison table | Results (§6) |
| `snr_estimator_check.png` | estimator accuracy over a ramp | If asked about the estimator |
| `eye_diagram.png` | QPSK/16-QAM eye + margins | If asked about pulse-shaping |
| `fading_vs_awgn.png` | Rayleigh penalty sweep | If asked about fading |
| `summary_report.md` | thresholds + corroboration + availability + headline | Your cheat sheet |

### E. Glossary (for your own confidence)

- **BER** — bit error rate: fraction of transmitted bits decoded incorrectly.
  Target here: **1e-3** (1 in 1000).
- **Eb/N0 vs Es/N0** — energy per bit vs per symbol, both over noise density.
  `Es/N0 (dB) = Eb/N0 (dB) + 10·log10(k)`. All decision thresholds and the SNR
  estimator work in **Es/N0**.
- **k (bits/symbol)** — BPSK 1, QPSK 2, 16-QAM 4, 64-QAM 6.
- **Constellation** — the set of allowed symbol points; more points = more
  bits/symbol but closer spacing → need more SNR to separate them.
- **Gray coding** — adjacent points differ in one bit, so a symbol error costs
  one bit error.
- **RRC / pulse shaping / matched filter** — the transmitted signal is shaped
  (β=0.22, span 10, 8 samples/symbol) and filtered at the receiver so that
  *at the sampling instants* there is **zero inter-symbol interference (ISI)**.
- **AWGN** — additive white Gaussian noise, the baseline impairment.
- **Rayleigh fading** — amplitude wobble from multipath (not shown live today,
  but in `fading_vs_awgn.png`).
- **Dwell** — one observation/decision slot of the link (40 in the trace).
- **SNR estimator (data-aided)** — correlates received pilot symbols against
  the known ones to estimate Es/N0 per dwell.
- **Hysteresis** — two thresholds per boundary so the mode does not oscillate
  when the SNR sits near a crossing.
- **Availability** — % of dwells where BER stays below target (link usable).
- **On-air vs off-air** — on-air = throughput active (BER < 1e-3); off-air =
  throughput forced to 0.
- **Monte-Carlo simulation** — estimating BER by actually transmitting random
  bits with noise and counting errors; the project's ground truth (seed 42,
  min 30 errors, ≤ 2 Mbit per point → ~15% precision).

### F. Likely questions & answers

**Q: Why doesn't the link always use 64-QAM?**
At low SNR its BER explodes — it needs ~22.8 dB Es/N0 to stay under 1e-3,
whereas BPSK needs ~7 dB. Always-64-QAM is off-air for 95% of this trace.

**Q: Why not always use BPSK, the "safest" one?**
BPSK has the best availability (47.5%) but the worst rate — 0.47 Mbit/s mean.
Adaptive gets near-BPSK availability with 2.5× the throughput.

**Q: What happens without hysteresis?**
A stateless selector oscillates between two modes when the SNR hovers near a
boundary (shown by the machine: alternating QPSK/16-QAM on every other call).
Hysteresis gives a 1 dB dead band: exactly 0–1 switches in a 100-step
oscillation test.

**Q: Why are thresholds in Es/N0, not Eb/N0?**
Because the receiver-side SNR estimator measures Es/N0 (from pilots). Per-bit
values are Es/N0 − 10·log10(k). The sweep itself ran on Eb/N0 then got
converted.

**Q: How are the thresholds derived?**
From a simulated BER sweep (symbol-level chain, seed 42, min-errors/bit-budget
protocol). The threshold of a scheme is the first SNR where the whole rest of
the sweep stays below 1e-3. They are re-verified at each switching point
(`verify_thresholds.py`).

**Q: Is the demo's BER as trustworthy as the report's?**
The demo uses the closed-form BER for instant updates; those closed forms were
validated against the Monte-Carlo chain (symbol-level and pulse-shaped
regression checks). The report's
numbers all come from the Monte-Carlo chain.

**Q: Why do availability/throughput differ between demo (47.5%, 50 Mbit) and
report (42.5%, 47.2 Mbit)?**
Same trace, same selector — only the BER estimator differs (closed form vs
noisy Monte-Carlo measurement). Dwells sitting within ~0.1 dB of a boundary
flip between "OK" and "OUT". A fraction-of-a-dB effect.

**Q: What does "greedy up, one-level down" mean?**
The selector may jump straight to the highest affordable mode when SNR is
great (e.g., straight to 64-QAM at start), but steps down only one level at a
time — a momentary wobble shouldn't dump you to the floor.

**Q: How good is the SNR estimator?**
Pilot-based, unit-normalized: bias +0.07 dB, RMS 0.33 dB, max error 0.67 dB
over a 2→28 dB ramp. Good enough that mode-selection errors only occur within
~0.5 dB of a boundary.

**Q: Where does "moving out of range" come from?**
A model, not live hardware: distance grows linearly (constant speed), log
-distance path loss 10·n·log10(d) with n = 2.2, plus ±0.4 dB Gaussian wobble
per dwell. Same trace for every chain so the comparison is apples-to-apples.

**Q: What about Rayleigh fading?**
Verified separately: penalties of +8.3 to +9.6 dB (BER = 1e-2) across schemes,
with ideal AGC at the receiver. It is a scenario knob, orthogonal to the
adaptive-MC machinery; the live demo runs the AWGN trace for speed.

### G. Troubleshooting & backup plan

- **No window appears** → backend is Agg (headless). Run
  `demo.py --fast --save`; it prints everything and writes the final frame.
  The narration works identically from the terminal alone.
- **Demo too fast to narrate** → raise `--rt` (e.g., 0.8).
- **Want a shorter run** → `--dwells 20` for a condensed pass.
- **Audience wants the true Monte-Carlo results** → run
  `main.py --adaptive-vs-fixed` beforehand (takes ~1–2 min); the printed table
  and `availability.csv` are the report numbers.
- **Anything crashes mid-demo** → Ctrl-C; a running summary is always printed,
  and every figure needed for the talk already exists in `results/` — the demo
  window is a luxury, not a dependency.
- **Warm-up** → run the demo once before the audience arrives (backend/font
  initialization) so the live run starts instantly.