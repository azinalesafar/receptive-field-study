"""
PBE-only and PBE+D3 DFT reference E_ads, built from the existing PBE+D4 DFT results.

The DFT energies come from SumCalculator([GPAW(PBE), DFTD4(method="PBE")]), and D4 is
a post-hoc sum, so for each DFT geometry (as in ../DFT/compute_adsorption_energies.py):

  E_PBE    = E_PBE+D4 - D4(geometry)          D4 with the SAME pbc/cell the DFT run used
  E_PBE+D3 = E_PBE + D3(BJ, PBE, two-body, full range; torch-dftd, same settings as the
                        m01nod4_d3 model variant)

  check: E_PBE must equal the last "Extrapolated:" energy in the GPAW log of that run
         (the value ASE returns for GPAW).

Pieces: relaxed system (_system_relaxed.xyz), substrate single point (_sp.extxyz),
relaxed isolated adsorbate (pbc xyz box 35.6 x 36.1 x 19.8 A, as GPAW PW needs).
Only approximation: geometries are PBE+D4-relaxed, not re-relaxed with PBE or PBE+D3.

The molecule's D3/D4 is evaluated in the same periodic box as its DFT run, mirroring how
the PBE+D4 reference was made; the non-periodic alternative is printed for comparison.

Appends rows with variant "DFT_PBE" and "DFT_PBE_D3" to a separate CSV,
dft_references.csv (same columns as eads_all_variants.csv).
Run in: module load gpaw/25.7.0; source .../SPICE_MACE/mace_env_cpu_disp/bin/activate
"""
import csv
import os
import re

import numpy as np
import torch
from ase.io import read
from dftd4.ase import DFTD4
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

HERE = os.path.dirname(os.path.abspath(__file__))
DFT = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT"
REF = "m-8_n0_m-6_n-8"
OUT = os.path.join(HERE, "dft_references.csv")


def d4(atoms):
    a = atoms.copy()
    a.calc = DFTD4(method="PBE")
    return a.get_potential_energy()


def d3(atoms):
    a = atoms.copy()
    a.calc = TorchDFTD3Calculator(xc="pbe", damping="bj", abc=False, device="cpu",
                                  dtype=torch.float64)
    return a.get_potential_energy()


def last_extrapolated(log):
    vals = re.findall(r"^Extrapolated:\s+(-?\d+\.\d+)", open(log).read(), re.M)
    return float(vals[-1]) if vals else None


def piece(xyz, log):
    a = read(xyz)
    e_tot = a.get_potential_energy()
    e_d4 = d4(a)
    e_pbe = e_tot - e_d4
    e_log = last_extrapolated(log)
    return dict(atoms=a, e_tot=e_tot, e_d4=e_d4, e_pbe=e_pbe, e_d3=d3(a),
                log_diff=None if e_log is None else e_pbe - e_log)


def main():
    dft = {r["label"]: r for r in csv.DictReader(open(f"{DFT}/adsorption_energies.csv"))}
    labels = sorted(dft, key=lambda l: float(dft[l]["e_ads"]))

    md = f"{DFT}/isolated_adsorbate/relaxed_results/isolated_adsorbate"
    mol = piece(f"{md}/isolated_adsorbate_system_relaxed.xyz", f"{md}/isolated_adsorbate_fin.log")
    iso = mol["atoms"].copy()
    iso.pbc = False
    print(f"molecule: pbc {mol['atoms'].pbc}  D4 {mol['e_d4']:.6f}  D3 {mol['e_d3']:.6f} eV  "
          f"(non-periodic: D4 {d4(iso):.6f}, D3 {d3(iso):.6f})  PBE-log {mol['log_diff']:+.2e} eV")

    rows, worst = [], 0.0
    for l in labels:
        sd = f"{DFT}/unitcell_candidates/DFT_relaxed/{l}"
        bd = f"{DFT}/substrate/sp_results/{l}"
        s = piece(f"{sd}/{l}_system_relaxed.xyz", f"{sd}/{l}_fin.log")
        b = piece(f"{bd}/{l}_sp.extxyz", f"{bd}/{l}_sp.log")
        e_d4ref = s["e_tot"] - b["e_tot"] - mol["e_tot"]
        if abs(e_d4ref - float(dft[l]["e_ads"])) > 1e-6:
            raise RuntimeError(f"{l}: PBE+D4 E_ads does not reproduce the DFT CSV")
        e_pbe = s["e_pbe"] - b["e_pbe"] - mol["e_pbe"]
        e_d3 = e_pbe + s["e_d3"] - b["e_d3"] - mol["e_d3"]
        for p, name in [(s, "system"), (b, "substrate")]:
            if p["log_diff"] is not None:
                worst = max(worst, abs(p["log_diff"]))
        print(f"{l:18s} E_ads  PBE+D4 {e_d4ref:+.4f}  PBE {e_pbe:+.4f}  PBE+D3 {e_d3:+.4f} eV   "
              f"log check sys {s['log_diff']:+.1e} sub {b['log_diff']:+.1e}")
        rows.append(dict(label=l, pbe=e_pbe, pbe_d3=e_d3))
    print(f"\nworst |E_PBE(strip) - E_PBE(GPAW log)| over systems/substrates: {worst:.2e} eV")

    ref = {k: next(r[k] for r in rows if r["label"] == REF) for k in ("pbe", "pbe_d3")}
    out = []
    for k, v in [("pbe", "DFT_PBE"), ("pbe_d3", "DFT_PBE_D3")]:
        for r in rows:
            out.append(dict(variant=v, label=r["label"], e_ads=r[k], de_ads=r[k] - ref[k],
                            converged=True, fragments=int(dft[r["label"]]["fragments"]),
                            min_d=float(dft[r["label"]]["min_intramolecular_distance"])))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
