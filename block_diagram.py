# SPDX-License-Identifier: MIT
# Copyright (c) 2026 georgian3768 and the Adaptive Modulation contributors

"""Clean, labeled block diagram of the adaptive-modulation link.

Chain drawn left-to-right:
    data bits -> Modulator -> RRC pulse shaping -> Channel (AWGN + fading)
              -> RRC matched filter -> SNR estimator -> Demodulator -> BER,
with the feedback arrow looping BACK from the SNR estimator to the Modulator
(the mode chosen for the next dwell follows adapt.py's threshold table with
its 1 dB hysteresis).

Saves results/link_block_diagram.svg (vector, report-ready) and .png.
Run:  .venv/bin/python block_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# palette
TX = "#dbe7f4"      # transmit-side (modulator, pulse shaping)
CH = "#fcefc2"      # channel
RX = "#e2f0da"      # receive-side (matched filter, estimator, demodulator)
BORDER = "#444444"

CY = 42.0            # main-chain y
H, W = 15.0, 12.0    # box size


def _box(ax, cx, cy, title, sub, face):
    ax.add_patch(FancyBboxPatch(
        (cx - W / 2, cy - H / 2), W, H,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        linewidth=1.3, edgecolor=BORDER, facecolor=face, zorder=2))
    ax.text(cx, cy + H * 0.12, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(cx, cy - H * 0.14, sub, ha="center", va="center",
            fontsize=7.8, zorder=3)


def _arrow(ax, x0, y0, x1, y1, rad=0.0, head=True):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>" if head else "-",
        mutation_scale=16, linewidth=1.4, color="#222222", zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(16.0, 7.2))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(56, 57, "Adaptive-modulation link \u2014 block diagram",
            ha="center", va="center", fontsize=13, fontweight="bold")

    centers = [13, 29, 45, 61, 77, 93]          # modulator .. demodulator
    _box(ax, 13, CY, "Modulator", "modulate()\nGray-coded, Es = 1\n"
                                   "{BPSK QPSK 16-QAM 64-QAM}", TX)
    _box(ax, 29, CY, "RRC pulse shaping", "\u03b2 = 0.22, span = 10\nsps = 8",
         TX)
    _box(ax, 45, CY, "Channel", "AWGN (N0/2 per dim)\n+ flat Rayleigh fading",
         CH)
    _box(ax, 61, CY, "RRC matched filter", "conv + sampler\nno ISI at t = nT",
         RX)
    _box(ax, 77, CY, "SNR estimator", "pilot block\nEs/N0 per dwell (dB)", RX)
    _box(ax, 93, CY, "Demodulator", "nearest-point decisions\noutput hard "
         "bits", RX)

    # input node and output node (no boxes)
    ax.text(2.6, CY, "data\nbits", ha="center", va="center", fontsize=9,
            style="italic", zorder=3)
    ax.text(106, CY, "per-dwell\nBER &\nthroughput", ha="center",
            va="center", fontsize=9, style="italic", zorder=3)

    # main chain arrows + small captions under them
    for x0, x1, cap in [(18.4, 23.6, "symbols"),
                        (34.4, 39.6, "shaped I/Q (sps = 8)"),
                        (50.4, 55.6, "received"),
                        (66.4, 71.6, "soft symbols"),
                        (82.4, 87.6, "hard bits")]:
        _arrow(ax, x0, CY, x1, CY)
        ax.text((x0 + x1) / 2, CY - 6.5, cap, ha="center", va="center",
                fontsize=8, style="italic", color="#333333")
    _arrow(ax, 98.4, CY, 103.2, CY)             # demod -> BER tally

    # feedback: SNR estimator bottom -> modulator bottom (loop under the row)
    _arrow(ax, 77, CY - H / 2, 77, 16, head=False)
    _arrow(ax, 77, 16, 13, 16, head=False)
    _arrow(ax, 13, 16, 13, CY - H / 2)
    ax.text(45, 22.0, "Es/N0 estimate \u2192 adapt.py select (threshold table, "
            "1 dB hysteresis \u2192 up/down shifts)",
            ha="center", va="center", fontsize=9.2, fontweight="bold",
            color="#1a5e5e")
    ax.text(45, 18.6, "next mode: {BPSK, QPSK, 16-QAM, 64-QAM}   "
            "(mode latency: picks for the following dwell)",
            ha="center", va="center", fontsize=8, color="#1a5e5e")

    ax.text(2, 5.5,
            "Es = 1 (unit-energy constellation); noise N0/2 per dimension per "
            "sample; RRC filters unit-norm; coherent reception, ideal AGC under "
            "fading. Only the SNR estimator feeds back - demodulation stays a "
            "nearest-point decision on the fixed lattice.",
            fontsize=7.5, color="#666666")

    plt.tight_layout()
    out_png = "results/link_block_diagram.png"
    out_svg = "results/link_block_diagram.svg"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)
    print(f"Block diagram saved to {out_png} and {out_svg}")


if __name__ == "__main__":
    main()