"""
Interaction energy vs height for DFT and the three model11-family variants (see README.md).

  E_int(dh) = E_system(dh) - E_substrate - E_molecule(frozen, same cell)   all single points

x axis = shortest molecule-graphene interatomic distance d_min. The learned model (r_max 5 A,
2 interactions) has no edge across the gap once d_min > 5 A, so its E_int must be exactly 0 there.
Writes eint_height_scan.csv and eint_height_scan.png; DFT points appear once the DFT jobs finish.
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
N_ADS, R_MAX = 126, 5.0
METHODS = [("dft", "DFT (PBE+D4)", "#0b0b0b", "o", "-"),
           ("mace/m11_d4trained", "learned vdW (10 Å)", "#2a78d6", "s", "-"),
           ("mace/m11nod4_d4", "post-hoc D4", "#eb6834", "^", "--"),
           ("mace/m11nod4", "no vdW", "#1baf7a", "D", ":")]


def E(p):
    return read(p).get_potential_energy() if os.path.exists(p) else None


def main():
    shifts = sorted(float(re.search(r"dh([\d.]+)\.extxyz", f).group(1))
                    for f in glob.glob(os.path.join(HERE, "geometries", "scan_dh*.extxyz")))
    dmin = {}
    for dh in shifts:
        a = read(os.path.join(HERE, "geometries", f"scan_dh{dh:.1f}.extxyz"))
        n = len(a) - N_ADS
        dmin[dh] = min(a.get_distances(i, range(n), mic=True).min() for i in range(n, len(a)))

    res = {}
    for d, label, *_ in METHODS:
        eb, em = E(f"{HERE}/{d}/substrate.extxyz"), E(f"{HERE}/{d}/molecule.extxyz")
        res[d] = {}
        for dh in shifts:
            es = E(f"{HERE}/{d}/scan_dh{dh:.1f}.extxyz")
            res[d][dh] = None if None in (es, eb, em) else es - eb - em

    with open(os.path.join(HERE, "eint_height_scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dh", "d_min"] + [lab for _, lab, *_ in METHODS])
        for dh in shifts:
            w.writerow([dh, f"{dmin[dh]:.3f}"] + ["" if res[d][dh] is None else f"{res[d][dh]:.6f}"
                                                  for d, *_ in METHODS])

    print("E_int (meV)")
    print(f"{'dh':>5s}{'d_min':>7s}" + "".join(f"{lab:>20s}" for _, lab, *_ in METHODS))
    for dh in shifts:
        print(f"{dh:5.1f}{dmin[dh]:7.2f}" + "".join(
            f"{1000 * res[d][dh]:20.1f}" if res[d][dh] is not None else f"{'-':>20s}" for d, *_ in METHODS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    x = np.array([dmin[dh] for dh in shifts])
    for ax, zoom in zip(axes, (False, True)):
        ax.set_facecolor("#fcfcfb")
        for d, label, col, mk, ls in METHODS:
            y = np.array([np.nan if res[d][dh] is None else res[d][dh] for dh in shifts])
            if np.all(np.isnan(y)):
                continue
            y = 1000 * y if zoom else y
            ax.plot(x, y, color=col, marker=mk, ls=ls, lw=2, ms=8, mec="#fcfcfb", mew=1.5, label=label)
        ax.axvline(R_MAX, color="#52514e", lw=1, ls=(0, (4, 3)))
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel("shortest molecule–graphene distance (Å)", color="#52514e")
        ax.tick_params(colors="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
    axes[0].set_ylabel("E_int (eV)", color="#52514e")
    axes[0].set_title("Interaction energy vs height", color="#0b0b0b", loc="left")
    axes[0].text(R_MAX + 0.1, axes[0].get_ylim()[0] * 0.95, "model cutoff\nr_max = 5 Å",
                 color="#52514e", fontsize=8, va="bottom")
    axes[1].set_xlim(R_MAX - 1.2, x.max() + 0.3)
    sel = x >= R_MAX - 1.2
    ys = [1000 * res[d][dh] for d, *_ in METHODS for dh, s in zip(shifts, sel) if s and res[d][dh] is not None]
    axes[1].set_ylim(min(ys) * 1.15 - 5, max(max(ys), 0) + 15)
    axes[1].set_ylabel("E_int (meV)", color="#52514e")
    axes[1].set_title("Tail beyond the cutoff (zoom)", color="#0b0b0b", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "eint_height_scan.png"), dpi=200)


if __name__ == "__main__":
    main()
