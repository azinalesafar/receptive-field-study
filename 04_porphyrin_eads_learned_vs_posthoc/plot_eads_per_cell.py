"""
Per-cell view of the model11 dispersion comparison: every one of the 26 cells gets its own
x position, so no point hides another (the S18-style parity plot stacks them into two clusters).

  top:    E_ads per cell (cells sorted by DFT E_ads); DFT as the reference marker, the three
          models side by side next to it
  bottom: model - DFT per cell (meV) for the two dispersion-handling models; the no-dispersion
          model (~+3.7 eV) is off this scale and shown only in the top panel

Usage: python3 plot_eads_per_cell.py   -> eads_per_cell_m11.png (and a copy in report_m11/)
"""
import csv
import os
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = [  # same colours/markers as plot_eads_parity_s18.py
    ("m11nod4", "no dispersion (model11_no_d4)", "#1baf7a", "s"),
    ("m11_d4trained", "dispersion learned from training data (model11)", "#2a78d6", "o"),
    ("m11nod4_d4", "dispersion added post hoc (model11_no_d4 + D4)", "#eb6834", "^"),
]
OFFSETS = {"DFT": 0.0, "m11nod4": -0.22, "m11_d4trained": 0.15, "m11nod4_d4": 0.30}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    e = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "eads_all_variants.csv"))):
        e[r["variant"]][r["label"]] = float(r["e_ads"])
    labels = sorted(e["DFT"], key=e["DFT"].get)
    x = np.arange(len(labels))
    dft = np.array([e["DFT"][l] for l in labels])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, facecolor=SURFACE,
                                 gridspec_kw=dict(height_ratios=[1.25, 1]))
    style(a1)
    style(a2)
    # alternating cell bands make each cell's group of markers easy to read
    for ax in (a1, a2):
        for i in x[::2]:
            ax.axvspan(i - 0.5, i + 0.5, color=GRID, alpha=0.35, linewidth=0, zorder=0)

    a1.scatter(x + OFFSETS["DFT"], dft, marker="D", s=30, color=INK, zorder=4, label="DFT (PBE+D4)")
    for v, name, colour, marker in SERIES:
        y = np.array([e[v][l] for l in labels])
        a1.scatter(x + OFFSETS[v], y, marker=marker, s=34, color=colour, edgecolor=SURFACE,
                   linewidth=0.8, zorder=3, label=f"{name}, MAE {np.abs(y - dft).mean():.2f} eV")
        if v != "m11nod4":
            a2.scatter(x + OFFSETS[v], 1000 * (y - dft), marker=marker, s=38, color=colour,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3, label=name)
    a1.set_ylabel("adsorption energy (eV)", color=INK)
    a1.set_title("Adsorption energy of each candidate cell: DFT vs three ways of handling dispersion",
                 color=INK, fontsize=11)
    leg = a1.legend(loc="center right", fontsize=8, frameon=True, labelcolor=INK2)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(SURFACE)

    a2.axhline(0, color=BASELINE, linewidth=1.2, zorder=1)
    a2.set_ylabel("model − DFT (meV)", color=INK)
    a2.text(0.01, 0.04, "no-dispersion model not shown here: its errors (~+3700 meV) are off this scale",
            transform=a2.transAxes, fontsize=8, color=MUTED)
    leg = a2.legend(loc="center left", fontsize=8, frameon=True, labelcolor=INK2)  # empty band, -300..-600
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(SURFACE)
    a2.set_xticks(x)
    a2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5, color=INK2)
    a2.set_xlim(-0.6, len(labels) - 0.4)
    a2.set_xlabel("candidate cell (sorted by DFT adsorption energy)", color=INK)

    fig.tight_layout()
    out = os.path.join(HERE, "eads_per_cell_m11.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    if os.path.isdir(os.path.join(HERE, "report_m11")):
        shutil.copy(out, os.path.join(HERE, "report_m11"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
