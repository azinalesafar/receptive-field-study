"""
Single S18-style parity plot (MAD-SURF SI Fig. S18): absolute E_ads of the 26 candidate cells,
three ways of handling dispersion, all against the same DFT reference (PBE+D4).

  no dispersion      model_01_no_d4                      (PBE-trained, nothing added)
  learned (training) model_01                            (trained on PBE+D4 labels)
  post-hoc           model_01_no_d4 + D4 at inference    (D4 in the relaxation, = DFT's D4)

Usage: python3 plot_eads_parity_s18.py [m01|m11] [--small [--inset]]   (m11 = model11 family; --small adds the
small-molecule-on-graphene points from ../small_molecules/eads_small_molecules.csv). All cells are drawn
the same way (no seen/held-out distinction, by request 2026-10-20; the report tabulates it). Reads eads_all_variants.csv (compute_eads.py).
Writes eads_parity_s18_style.png (m01) or eads_parity_s18_style_m11.png.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SMALL_CSV = os.path.join(os.path.dirname(HERE), "small_molecules", "eads_small_molecules.csv")

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# categorical slots 1-3 of the reference palette (validated all-pairs for scatter), plus shapes
FAMILIES = {
    "m01": [  # variant, legend label, colour, marker
        ("m01nod4", "no dispersion (model_01_no_d4)", "#1baf7a", "s"),
        ("m01_d4trained", "dispersion learned from training data (model_01, PBE+D4)", "#2a78d6", "o"),
        ("m01nod4_d4", "dispersion added post hoc (model_01_no_d4 + D4)", "#eb6834", "^"),
    ],
    "m11": [
        ("m11nod4", "no dispersion (model11_no_d4)", "#1baf7a", "s"),
        ("m11_d4trained", "dispersion learned from training data (model11, PBE+D4)", "#2a78d6", "o"),
        ("m11nod4_d4", "dispersion added post hoc (model11_no_d4 + D4)", "#eb6834", "^"),
    ],
}
# cells whose DFT frames were used to fine-tune model11 / model11_no_d4 (train + valid splits of
# SPICE_MACE/model11_adsorption_dft_activelearning/dataset); used by make_report_m11.py
M11_SEEN = {"m0_n-8_m8_n2", "m0_n-8_m-8_n-6", "m-4_n5_m6_n8", "m-5_n4_m-8_n-6", "m-8_n0_m2_n8",
            "m8_n0_m-2_n-8", "m-8_n0_m-6_n-8", "m8_n0_m6_n8", "m-8_n-8_m2_n-6", "m-8_n-8_m-6_n2",
            "m8_n8_m6_n-2", "m5_n-4_m8_n6"}


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "m01"
    with_small = "--small" in sys.argv   # add the small-molecule-on-graphene points (m11 only)
    with_inset = with_small and "--inset" in sys.argv   # zoom on the small-molecule region
    SERIES = FAMILIES[family]
    e = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "eads_all_variants.csv"))):
        e[r["variant"]][r["label"]] = float(r["e_ads"])
    labels = sorted(e["DFT"], key=e["DFT"].get)
    n_cells = len(labels)
    if with_small:
        for r in csv.DictReader(open(SMALL_CSV)):
            e[r["variant"]]["sm:" + r["label"]] = float(r["e_ads"])
        labels += sorted(l for l in e["DFT"] if l.startswith("sm:"))
    x = np.array([e["DFT"][l] for l in labels])
    is_cell = np.array([not l.startswith("sm:") for l in labels])

    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = (-7.0, 0.6)
    ax.plot(lim, lim, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(-6.85, -6.55, "DFT = model", rotation=45, color=MUTED, fontsize=8, va="bottom")

    for v, name, colour, marker in SERIES:
        if len(e.get(v, {})) < len(labels):
            print(f"{v}: missing, not plotted")
            continue
        y = np.array([e[v][l] for l in labels])
        mae = np.abs(y - x).mean()
        lab = f"{name}\nMAE {mae:.2f} eV"
        if with_small:
            lab = (f"{name}\nMAE {np.abs(y - x)[is_cell].mean():.2f} eV (cells), "
                   f"{np.abs(y - x)[~is_cell].mean():.2f} eV (small molecules)")
        ax.scatter(x, y, s=40 if with_small else 46, marker=marker, color=colour, edgecolor=SURFACE,
                   linewidth=1.0, alpha=0.85, zorder=3, label=lab)
        print(f"{v:15s} MAE {mae:.3f} eV   mean E_ads {y.mean():+.3f} eV")
    print(f"{'DFT':15s}                 mean E_ads {x.mean():+.3f} eV")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("DFT adsorption energy, PBE+D4 (eV)", color=INK)
    ax.set_ylabel("MACE adsorption energy (eV)", color=INK)
    ax.set_title(f"Adsorption energy parity: {n_cells} candidate unit cells"
                 + (f" + {len(labels) - n_cells} small molecules on graphene" if with_small else ""),
                 color=INK, fontsize=10 if with_small else 11)
    leg = ax.legend(loc="lower right", fontsize=7.5, frameon=True, labelcolor=INK2,
                    borderpad=0.8, handletextpad=0.6)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(SURFACE)

    if with_inset:
        # zoom on the small-molecule region, placed in the empty middle-right of the main plot
        zlim = (-2.25, 0.15)
        ins = ax.inset_axes([0.585, 0.245, 0.37, 0.37])   # band between legend and the small-molecule cluster
        ins.set_facecolor(SURFACE)
        ins.plot(zlim, zlim, color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
        sm = ~is_cell
        for v, name, colour, marker in SERIES:
            if len(e.get(v, {})) < len(labels):
                continue
            y = np.array([e[v][l] for l in labels])
            ins.scatter(x[sm], y[sm], s=22, marker=marker, color=colour, edgecolor=SURFACE,
                        linewidth=0.6, alpha=0.85, zorder=3)
            k = np.polyfit(x[sm], y[sm], 1)[0]
            ins.plot(zlim, np.polyval(np.polyfit(x[sm], y[sm], 1), zlim), color=colour,
                     linewidth=1.0, alpha=0.8, zorder=2)
        ins.set_xlim(zlim)
        ins.set_ylim(zlim)
        ins.set_aspect("equal")
        ins.grid(color=GRID, linewidth=0.6, zorder=0)
        ins.set_axisbelow(True)
        for sp in ins.spines.values():
            sp.set_color(BASELINE)
        ins.tick_params(colors=MUTED, labelsize=7)
        ins.set_title("small molecules (lines: least-squares fits)", fontsize=7.5, color=INK2, pad=3)
        ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.6)
    fig.tight_layout()
    out = os.path.join(HERE, "eads_parity_s18_style.png" if family == "m01" else
                       f"eads_parity_s18_style_{family}{'_all' if with_small else ''}{'_inset' if with_inset else ''}.png")
    fig.savefig(out, dpi=220, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
