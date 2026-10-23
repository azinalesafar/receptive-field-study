#!/usr/bin/env python3
"""
Missing E_ads vs cutoff radius, grouped by lattice cluster.

Reads the finished results of measure_eads_tail_vs_R.py (eads_tail_vs_R.npz);
no D4 is run. Clusters are the distinct lattices (L1, L2, angle) in
ads_eng_calc/adsorption_energy_comparison_grouped.csv; structures not listed
there are drawn in grey. Thin lines are individual structures, thick lines the
cluster mean.

    tail_ads(R) = t_system(R) - t_substrate(R) - t_adsorbate(R)

--isolated-molecule uses the non-periodic adsorbate for t_adsorbate. The
default uses the adsorbate as stored in its file (periodic in x, y, z, as in the
DFT reference); its own image 19.8 Ang away in z produces a bump at R ~ 20 Ang.
The term is identical for every structure, so the choice shifts all curves
together and does not change any difference between them.

USAGE
    python3 plot_eads_tail_by_cluster.py
    python3 plot_eads_tail_by_cluster.py --isolated-molecule
"""
import os
import csv
import argparse

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(THIS_DIR, "eads_tail_vs_R.npz")
GROUPED = "/scratch/johnt447/azin/analysis/ads_eng_calc/adsorption_energy_comparison_grouped.csv"
DFT = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT/adsorption_energies.csv"
RECEPTIVE_FIELD = 10.0

# Okabe-Ito, assigned in order of DFT stability (most stable first).
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
UNGROUPED = "#888888"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolated-molecule", action="store_true",
                    help="use the non-periodic adsorbate for the adsorbate term")
    args = ap.parse_args()

    a = np.load(NPZ)
    radii, tags = a["radii"], [str(t) for t in a["tags"]]
    t_mol = a["t_mol_iso"] if args.isolated_molecule else a["t_mol"]
    tail = a["t_sys"] - a["t_sub"] - t_mol[None, :]

    lattice = {r["label"]: (round(float(r["L1_A"]), 2), round(float(r["L2_A"]), 2),
                            round(float(r["angle_deg"]), 1))
               for r in csv.DictReader(open(GROUPED))}
    dft = {r["label"]: 1000.0 * float(r["e_ads"]) for r in csv.DictReader(open(DFT))}

    clusters = {}
    for k, t in enumerate(tags):
        clusters.setdefault(lattice.get(t), []).append(k)
    grouped = sorted((c for c in clusters if c is not None),
                     key=lambda c: np.mean([dft[tags[k]] for k in clusters[c]]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    entries = [(c, COLOURS[j % len(COLOURS)]) for j, c in enumerate(grouped)]
    if None in clusters:
        entries.append((None, UNGROUPED))
    for c, colour in entries:
        idx = clusters[c]
        for k in idx:
            ax.plot(radii, tail[k], "-", lw=0.7, color=colour, alpha=0.35)
        if c is None:
            label = "not in grouped CSV"
        else:
            e_mean = np.mean([dft[tags[k]] for k in idx])
            label = (f"{c[0]:.2f} $\\AA$, {c[1]:.2f} $\\AA$, {c[2]:.1f}$\\degree$  "
                     f"(n={len(idx)}, DFT mean {e_mean:.0f} meV)")
        ax.plot(radii, tail[idx].mean(axis=0), "-", lw=2.4, color=colour, label=label)

    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    ax.axvline(RECEPTIVE_FIELD, color="#D55E00", lw=1.6, ls="--", zorder=1)
    ax.set_xlabel(r"cutoff radius $R$ ($\AA$)")
    ax.set_ylabel(r"part of $E_{\rm ads}$ beyond $R$ (meV)")
    ax.set_title(r"Missing $E_{\rm ads}$ by lattice cluster")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    name = ("eads_tail_by_cluster_vs_R_isolated_molecule.png" if args.isolated_molecule
            else "eads_tail_by_cluster_vs_R.png")
    p = os.path.join(THIS_DIR, name)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"wrote {os.path.relpath(p, THIS_DIR)}")


if __name__ == "__main__":
    main()
