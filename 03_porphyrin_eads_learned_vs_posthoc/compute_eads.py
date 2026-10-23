"""
E_ads for every variant in tasks.txt, plus the DFT reference, into one table.

  E_ads    = E_system - E_substrate - E_adsorbate          (same as ../DFT/compute_adsorption_energies.py)
  dE_ads   = E_ads(label) - E_ads(REF)                      (isolated adsorbate cancels exactly)

REF is the DFT global minimum. Structural integrity (fragment count, intramolecular
min distance) and optimizer convergence are recorded per row, not silently dropped.

Writes eads_all_variants.csv (long format) and prints a per-variant status summary.
"""
import csv
import os
import sys

from ase.io import read

sys.path.insert(0, "/scratch/johnt447/azin/analysis/ads_eng_calc/MACE_model01")
from compute_adsorption_energies import n_fragments_and_min_dist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DFT_CSV = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT/adsorption_energies.csv"
REF = "m-8_n0_m-6_n-8"
N_ADS = 126
OUT = os.path.join(HERE, "eads_all_variants.csv")


def variants():
    seen = []
    for fn in sorted(f for f in os.listdir(HERE) if f.startswith("tasks") and f.endswith(".txt")):
        for line in open(os.path.join(HERE, fn)):
            v = line.split()[0]
            if v not in seen and not v.startswith("repro"):
                seen.append(v)
    return seen


def main():
    dft = {r["label"]: r for r in csv.DictReader(open(DFT_CSV))}
    labels = sorted(dft, key=lambda l: float(dft[l]["e_ads"]))
    rows = [dict(variant="DFT", label=l, e_ads=float(dft[l]["e_ads"]),
                 converged=True, fragments=int(dft[l]["fragments"]),
                 min_d=float(dft[l]["min_intramolecular_distance"])) for l in labels]

    for v in variants():
        mol_f = os.path.join(HERE, v, "molecule", "isolated_adsorbate.extxyz")
        if not os.path.exists(mol_f):
            print(f"{v}: isolated adsorbate missing, skipped")
            continue
        mol = read(mol_f)
        e_mol = mol.get_potential_energy()
        missing, notconv, flagged = [], [], []
        for l in labels:
            sf = os.path.join(HERE, v, "system", f"{l}.extxyz")
            bf = os.path.join(HERE, v, "substrate", f"{l}.extxyz")
            if not (os.path.exists(sf) and os.path.exists(bf)):
                missing.append(l)
                continue
            sys_a, sub_a = read(sf), read(bf)
            frags, min_d = n_fragments_and_min_dist(sys_a[len(sys_a) - N_ADS:])
            conv = bool(sys_a.info.get("converged")) and bool(mol.info.get("converged"))
            if not conv:
                notconv.append(l)
            if frags > 1 or min_d < 0.9:
                flagged.append(l)
            rows.append(dict(variant=v, label=l,
                             e_ads=sys_a.get_potential_energy() - sub_a.get_potential_energy() - e_mol,
                             converged=conv, fragments=frags, min_d=min_d))
        print(f"{v}: {len(labels) - len(missing)}/{len(labels)} done; "
              f"not converged {notconv or '-'}; integrity flags {flagged or '-'}; "
              f"molecule converged {mol.info.get('converged')}")

    # dE_ads relative to REF, per variant
    ref = {r["variant"]: r["e_ads"] for r in rows if r["label"] == REF}
    for r in rows:
        r["de_ads"] = r["e_ads"] - ref[r["variant"]] if r["variant"] in ref else None

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "label", "e_ads", "de_ads",
                                          "converged", "fragments", "min_d"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
