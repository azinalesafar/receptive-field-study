"""
Paper version of the adsorption-energy parity plot (26 candidate cells + 57 small molecules on
graphene, model11 family vs DFT PBE+D4). Same data as
../dispersion_inference/eads_parity_s18_style_m11_all[_inset].png; only the presentation differs:
title, axis label, legend names, no grid, inset without its title.

Writes, in this folder, PDF (vector) and PNG (300 dpi):
  adsorption_energy_parity.{pdf,png}         without the zoomed-in section
  adsorption_energy_parity_zoom.{pdf,png}    with the zoomed-in small-molecule section

Run: module load gpaw/25.7.0; source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
     python3 plot_paper_parity.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CELLS = os.path.join(ROOT, "dispersion_inference", "eads_all_variants.csv")
SMALL = os.path.join(ROOT, "small_molecules", "eads_small_molecules.csv")

INK, INK2, MUTED, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#ffffff"
SERIES = [  # variant, legend name, colour, marker (colours/markers as in the working plots)
    ("m11nod4", "fine_tuned_MACE_no_vdw", "#1baf7a", "s"),
    ("m11_d4trained", "fine_tuned_MACE_vdw_learned", "#2a78d6", "o"),
    ("m11nod4_d4", "fine_tuned_MACE_post_hoc_vdw", "#eb6834", "^"),
]
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})  # editable text in PDF


def load():
    e = {}
    for r in csv.DictReader(open(CELLS)):
        e.setdefault(r["variant"], {})["cell:" + r["label"]] = float(r["e_ads"])
    for r in csv.DictReader(open(SMALL)):
        e.setdefault(r["variant"], {})["sm:" + r["label"]] = float(r["e_ads"])
    labels = sorted(l for l in e["DFT"] if all(l in e[v] for v, *_ in SERIES))
    return e, labels


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def make(zoom):
    e, labels = load()
    x = np.array([e["DFT"][l] for l in labels])
    is_cell = np.array([l.startswith("cell:") for l in labels])

    fig, ax = plt.subplots(figsize=(6.0, 6.0), facecolor=SURFACE)
    style(ax)
    lim = (-7.0, 0.6)
    ax.plot(lim, lim, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(-6.85, -6.55, "DFT = model", rotation=45, color=MUTED, fontsize=8, va="bottom")
    for v, name, colour, marker in SERIES:
        y = np.array([e[v][l] for l in labels])
        err = np.abs(y - x)
        ax.scatter(x, y, s=40, marker=marker, color=colour, edgecolor=SURFACE, linewidth=0.9,
                   alpha=0.9, zorder=3,
                   label=f"{name}\nMAE {err[is_cell].mean():.2f} eV (cells), {err[~is_cell].mean():.2f} eV (small molecules)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT adsorption energy (eV)", color=INK)
    ax.set_ylabel("MACE adsorption energy (eV)", color=INK)
    ax.set_title("Adsorption energy plot", color=INK, fontsize=12)
    leg = ax.legend(loc="lower right", fontsize=7.5, frameon=True, labelcolor=INK2,
                    borderpad=0.8, handletextpad=0.6)
    leg.get_frame().set_edgecolor(BASELINE)
    leg.get_frame().set_facecolor(SURFACE)

    if zoom:
        zlim = (-2.25, 0.15)
        ins = ax.inset_axes([0.585, 0.245, 0.37, 0.37])
        style(ins)
        for s in ["top", "right"]:
            ins.spines[s].set_visible(True)
            ins.spines[s].set_color(INK2)
        ins.plot(zlim, zlim, color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
        sm = ~is_cell
        for v, name, colour, marker in SERIES:
            y = np.array([e[v][l] for l in labels])
            ins.scatter(x[sm], y[sm], s=20, marker=marker, color=colour, edgecolor=SURFACE,
                        linewidth=0.5, alpha=0.9, zorder=3)
            ins.plot(zlim, np.polyval(np.polyfit(x[sm], y[sm], 1), zlim), color=colour,
                     linewidth=1.0, alpha=0.8, zorder=2)
        ins.set_xlim(zlim)
        ins.set_ylim(zlim)
        ins.set_aspect("equal")
        ins.tick_params(labelsize=7)
        ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.7)

    fig.tight_layout()
    base = os.path.join(HERE, "adsorption_energy_parity" + ("_zoom" if zoom else ""))
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {base}.pdf/.png")


if __name__ == "__main__":
    make(zoom=False)
    make(zoom=True)
