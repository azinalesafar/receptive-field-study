"""
Height-scan binding curve in the style of Fig. 4 of Solé et al., arXiv:2605.25710 (2026):
E_int vs distance for DFT and the three model variants, with an inset zoomed on the well.
Reads eint_height_scan.csv (written by compute_eint.py); the DFT curve appears once its jobs are done.

  python plot_fig4_style.py            E_int, molecule reference = frozen molecule in the slab cell
                                       -> eint_fig4_style.png
  python plot_fig4_style.py --isolated molecule reference = relaxed isolated molecule, i.e. the same
                                       E_ads convention as ../dispersion_inference/eads_per_cell_m11.png
                                       -> eads_profile_fig4_style.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

HERE = os.path.dirname(os.path.abspath(__file__))
R_MAX = 5.0
# column in the CSV, legend label, colour, marker, line width
SERIES = [("learned vdW (10 Å)", "learned vdW (10 Å)", "#2a78d6", "s", 2),
          ("post-hoc D4", "post-hoc D4", "#eb6834", "^", 2),
          ("no vdW", "no vdW", "#1baf7a", "D", 2),
          ("DFT (PBE+D4)", "DFT (PBE+D4)", "#9a9a9a", "o", 5)]
WELL = (2.0, 3.0)   # inset x range (Å)
D = "/scratch/johnt447/azin/analysis/ads_eng_calc"
# CSV column -> (height-scan result dir, relaxed isolated molecule used for E_ads)
ISOLATED = {
    "DFT (PBE+D4)": ("dft", f"{D}/DFT/isolated_adsorbate/relaxed_results/isolated_adsorbate/"
                            "isolated_adsorbate_system_relaxed.xyz"),
    "learned vdW (10 Å)": ("mace/m11_d4trained", f"{D}/dispersion_inference/m11_d4trained/molecule/isolated_adsorbate.extxyz"),
    "post-hoc D4": ("mace/m11nod4_d4", f"{D}/dispersion_inference/m11nod4_d4/molecule/isolated_adsorbate.extxyz"),
    "no vdW": ("mace/m11nod4", f"{D}/dispersion_inference/m11nod4/molecule/isolated_adsorbate.extxyz"),
}


def isolated_offset(col):
    """E_mol(frozen, slab cell) - E_mol(relaxed, isolated), in eV: added to E_int it gives E_ads(dh)."""
    from ase.io import read
    d, iso = ISOLATED[col]
    return (read(os.path.join(HERE, d, "molecule.extxyz")).get_potential_energy()
            - read(iso).get_potential_energy())


def dft_d4_part(rows, isolated):
    """D4 part of the DFT curve (eV): e_d4(system) - e_d4(substrate) - e_d4(molecule), using the D4
    energies stored by dft_calc.py. With --isolated the molecule term is the relaxed isolated
    molecule, whose file has no stored D4: it is computed once with dftd4 (needs module gpaw/25.7.0)
    and cached in dft/isolated_molecule_d4.json, so later runs work without the module."""
    import json
    from ase.io import read
    d = os.path.join(HERE, "dft")
    sub = read(os.path.join(d, "substrate.extxyz")).info["e_d4"]
    if isolated:
        cache = os.path.join(d, "isolated_molecule_d4.json")
        if os.path.exists(cache):
            with open(cache) as f:
                mol = json.load(f)["e_d4"]
        else:
            from dftd4.ase import DFTD4   # module load gpaw/25.7.0
            src = ISOLATED["DFT (PBE+D4)"][1]
            m = read(src)
            m.calc = DFTD4(method="PBE")
            mol = m.get_potential_energy()
            with open(cache, "w") as f:
                json.dump({"e_d4": mol, "source": src, "method": "DFTD4(method='PBE')"}, f, indent=1)
    else:
        mol = read(os.path.join(d, "molecule.extxyz")).info["e_d4"]
    y = []
    for r in rows:
        p = os.path.join(d, f"scan_dh{float(r['dh']):.1f}.extxyz")
        y.append(read(p).info["e_d4"] - sub - mol if os.path.exists(p) else np.nan)
    return np.array(y)


def main():
    isolated = "--isolated" in sys.argv
    with open(os.path.join(HERE, "eint_height_scan.csv")) as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r["d_min"]) for r in rows])
    data = {}
    for col, *_ in SERIES:
        y = np.array([float(r[col]) if r[col] else np.nan for r in rows])
        if not np.all(np.isnan(y)):
            data[col] = y + isolated_offset(col) if isolated else y

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ins = ax.inset_axes([0.50, 0.14, 0.45, 0.48])
    for col, label, color, mk, lw in SERIES:
        if col not in data:
            continue
        ok = ~np.isnan(data[col])
        z = 1 if col.startswith("DFT") else 2
        for a in (ax, ins):
            a.plot(x[ok], data[col][ok], color=color, lw=lw, marker=mk, ms=5 if a is ax else 6,
                   mec="white", mew=0.8, label=label, zorder=z)

    if "DFT (PBE+D4)" in data:
        yd4 = dft_d4_part(rows, isolated)
        ok = ~np.isnan(yd4)
        for a in (ax, ins):
            a.plot(x[ok], yd4[ok], color="#6f6f6f", lw=2, ls=(0, (5, 2)),
                   label="DFT, D4 part only", zorder=1.5)

    ax.axvline(R_MAX, color="#52514e", lw=1, ls=(0, (4, 3)), zorder=0)
    ax.text(R_MAX + 0.08, -2.7 if isolated else -0.6, "model cutoff\n(5 Å)", color="#52514e",
            fontsize=9, va="top")
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    ax.set_xlim(2.0, x.max() + 0.2)
    ax.set_xticks(np.arange(2, 12))
    ax.set_xlabel("Shortest molecule–graphene distance [Å]")
    ax.set_ylabel("Adsorption energy [eV]" if isolated else "Interaction energy [eV]")

    sel = (x >= WELL[0]) & (x <= WELL[1])
    ys = np.concatenate([data[c][sel] for c in data if c != "no vdW"])
    ys = ys[~np.isnan(ys)]
    ins.set_xlim(*WELL)
    ins.set_ylim(ys.min() - 0.15, ys.max() + 0.15)
    ins.set_xlabel("Distance [Å]", fontsize=9)
    ins.set_ylabel("Energy [eV]", fontsize=9)
    ins.tick_params(labelsize=8)
    mark_inset(ax, ins, loc1=1, loc2=3, fc="none", ec="#52514e", lw=0.8, ls="--")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True, fontsize=9)
    fig.tight_layout()
    stem = "eads_profile_fig4_style" if isolated else "eint_fig4_style"
    for ext in ("png",):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
