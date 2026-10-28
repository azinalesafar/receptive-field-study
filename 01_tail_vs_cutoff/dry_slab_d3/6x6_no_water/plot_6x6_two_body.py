#!/usr/bin/env python3
"""
Missing two-body D3(BJ) energy vs pair cutoff R for the 6x6 dry cell (9144 atoms).

Job 1877883 was cancelled before the 6x6 ATM calls finished, so only the two-body
curve exists. Its energies are read from the job log (read only); nothing is
recalculated. Same quantity and style as the 2x2 full_range_10-50A plot:

    missing(R) = (E(50.27 A) - E(R)) / N     [meV/atom], CN cutoff fixed at 21.17 A

Outputs (this directory): d3_vs_R_6x6_two_body.csv, d3_vs_R_6x6_two_body.png (10-50 A),
                          d3_vs_R_6x6_two_body_10-15A.png (10-15 A only)

Usage:  module load gpaw/25.7.0; python3 plot_6x6_two_body.py
"""

import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "d3_vs_R_1877883.out")
N = 9144
R_REF = 50.272                    # dftd3 default two-body cutoff (95 bohr)
R_CHECK = 80.0
RECEPTIVE_FIELD = 10.0            # MACE production receptive field

LINE = re.compile(r"^\s+2b\s+cut=\s*([\d.]+)\s+cn=\s*[\d.]+\s+abc=0\s+E=\s*(-?[\d.]+)")


def main():
    with open(LOG) as fh:
        lines = fh.read().splitlines()
    start = next(i for i, l in enumerate(lines) if "6x6" in l and "structure" in l)
    E = {}
    for line in lines[start:]:
        m = LINE.match(line)
        if m:
            E[float(m.group(1))] = float(m.group(2))

    radii = np.array(sorted(r for r in E if r < R_REF - 1e-6))
    e2 = np.array([E[r] for r in radii])
    miss2 = (E[R_REF] - e2) / N * 1000.0
    dref = (E[R_CHECK] - E[R_REF]) / N * 1000.0

    np.savetxt(os.path.join(HERE, "d3_vs_R_6x6_two_body.csv"),
               np.column_stack([radii, e2, miss2]), delimiter=",", fmt="%.8f",
               header="R_A,E_2body_eV,missing_2body_meV_per_atom", comments="")
    print(f"{len(radii)} radii {radii[0]:g}-{radii[-1]:g} A, reference E({R_REF}) = "
          f"{E[R_REF]:.6f} eV; E({R_CHECK:g}) - E(ref) = {dref:+.6f} meV/atom")
    for r, m in zip(radii, miss2):
        if r in (10, 12, 15, 20, 30):
            print(f"  R = {r:4.0f} A   missing = {m:.4f} meV/atom")

    make_plot(radii, miss2, "d3_vs_R_6x6_two_body.png")
    make_plot(radii, miss2, "d3_vs_R_6x6_two_body_10-15A.png", r_max=15.0)
    print("wrote d3_vs_R_6x6_two_body.png, d3_vs_R_6x6_two_body_10-15A.png / .csv")


def make_plot(radii, miss2, name, r_max=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if r_max is not None:
        keep = radii <= r_max + 1e-6
        radii, miss2 = radii[keep], miss2[keep]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(radii, miss2, "-s", ms=4, lw=1.8, color="#0072B2", label="D3(BJ) 2-body")
    rr = np.linspace(radii[0], radii[-1], 200)
    ax.plot(rr, miss2[0] * (radii[0] / rr) ** 4, ":", lw=1.4, color="#0072B2",
            label=r"$1/R^4$ from 2-body at %.0f $\AA$" % radii[0])
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    ax.axvline(RECEPTIVE_FIELD, color="#D55E00", lw=1.2, ls="--", alpha=0.5)
    ax.set_xlim(radii[0] - 0.3, radii[-1] + 0.3)
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_ylabel(r"missing dispersion $(E_{\rm ref}-E(R))/N$  (meV/atom)")
    ax.set_title(f"D3 energy beyond a global cutoff $R$ (6x6, {N} atoms, ref = {R_REF:.1f} $\\AA$)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
