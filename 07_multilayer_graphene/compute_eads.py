"""
E_ads on 1/2/3-layer graphene for DFT and the three model11-family variants (see README.md).

  E_ads = E_system(SP) - E_substrate(SP) - E_molecule(relaxed isolated)
  DFT 1L : existing candidate-cell DFT (system_relaxed, substrate SP), same geometry/box/settings
  DFT 2L/3L : dft/<cell>_<n>L_{system,substrate}.extxyz
  models : mace/<variant>/<cell>_<n>L_{system,substrate}.extxyz; molecule from ../dispersion_inference

Key quantity: the extra binding from the added layers, dE(nL) = E_ads(nL) - E_ads(1L), per method,
compared with DFT. The learned model only sees 10 A; post-hoc D4 reaches ~32 A.
Writes eads_multilayer.csv and prints the tables.
"""
import csv
import os

import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT"
DI = "/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference"
CELLS = ["m-8_n0_m-6_n-8", "m0_n8_m8_n6", "m-8_n-1_m-5_n-8", "m-8_n-6_m-5_n3"]
MODELS = [("m11nod4", "no_vdw"), ("m11_d4trained", "vdw_learned"), ("m11nod4_d4", "post_hoc_vdw")]


def E(p):
    return read(p).get_potential_energy() if os.path.exists(p) else None


def main():
    chk = E(os.path.join(HERE, "dft", "m-8_n0_m-6_n-8_1L_system.extxyz"))
    ref = E(f"{D}/unitcell_candidates/DFT_relaxed/m-8_n0_m-6_n-8/m-8_n0_m-6_n-8_system_relaxed.xyz")
    if chk is not None:
        print(f"settings check (1L system recomputed vs existing DFT): {1000 * (chk - ref):+.3f} meV\n")

    mol = {"DFT": E(f"{D}/isolated_adsorbate/relaxed_results/isolated_adsorbate/isolated_adsorbate_system_relaxed.xyz")}
    for v, _ in MODELS:
        mol[v] = E(f"{DI}/{v}/molecule/isolated_adsorbate.extxyz")

    rows = []
    for c in CELLS:
        for n in (1, 2, 3):
            if n == 1:
                es = E(f"{D}/unitcell_candidates/DFT_relaxed/{c}/{c}_system_relaxed.xyz")
                eb = E(f"{D}/substrate/sp_results/{c}/{c}_sp.extxyz")
            else:
                es = E(os.path.join(HERE, "dft", f"{c}_{n}L_system.extxyz"))
                eb = E(os.path.join(HERE, "dft", f"{c}_{n}L_substrate.extxyz"))
            row = dict(cell=c, layers=n, DFT=None if None in (es, eb) else es - eb - mol["DFT"])
            for v, _ in MODELS:
                ms = E(os.path.join(HERE, "mace", v, f"{c}_{n}L_system.extxyz"))
                mb = E(os.path.join(HERE, "mace", v, f"{c}_{n}L_substrate.extxyz"))
                row[v] = None if None in (ms, mb) else ms - mb - mol[v]
            rows.append(row)

    keys = ["DFT"] + [v for v, _ in MODELS]
    names = {"DFT": "DFT", **dict(MODELS)}
    with open(os.path.join(HERE, "eads_multilayer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "layers"] + keys)
        w.writeheader()
        w.writerows(rows)

    print("E_ads (eV)")
    print(f"{'cell':18s}{'L':>3s}" + "".join(f"{names[k]:>15s}" for k in keys))
    for r in rows:
        print(f"{r['cell']:18s}{r['layers']:3d}" + "".join(f"{r[k]:15.3f}" if r[k] is not None else f"{'-':>15s}" for k in keys))

    print("\nExtra binding from added layers, dE(nL) = E_ads(nL) - E_ads(1L) (meV); error vs DFT in brackets")
    for n in (2, 3):
        print(f"  {n} layers:")
        for k in keys:
            vals = []
            for c in CELLS:
                r1 = next(r for r in rows if r["cell"] == c and r["layers"] == 1)
                rn = next(r for r in rows if r["cell"] == c and r["layers"] == n)
                if r1[k] is not None and rn[k] is not None and r1["DFT"] is not None and rn["DFT"] is not None:
                    vals.append((rn[k] - r1[k], (rn[k] - r1[k]) - (rn["DFT"] - r1["DFT"])))
            if vals:
                d = np.array(vals)
                print(f"    {names[k]:13s} mean {1000 * d[:, 0].mean():+7.0f} meV"
                      + ("" if k == "DFT" else f"   [error {1000 * d[:, 1].mean():+6.0f} meV]"))

    print("\nMAE vs DFT per thickness (eV)")
    for n in (1, 2, 3):
        sel = [r for r in rows if r["layers"] == n and r["DFT"] is not None]
        print(f"  {n}L: " + "  ".join(
            f"{names[v]} {np.mean([abs(r[v] - r['DFT']) for r in sel if r[v] is not None]):.3f}"
            for v, _ in MODELS if any(r[v] is not None for r in sel)))


if __name__ == "__main__":
    main()
