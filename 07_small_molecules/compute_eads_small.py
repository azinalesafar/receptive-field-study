"""
E_ads of the small-molecule-on-graphene runs (model_01 DFT data), DFT and the three model11-family
models, with the same definition as the candidate cells:

  E_ads = E_system - E_graphene - E_molecule
  DFT  : E_system = stored model_01 DFT end point (runs.csv), E_graphene = dft/graphene,
         E_molecule = dft/molecules (isolated relax, vacuum 8 A, Gamma)
  model: mace/<variant>/{system,substrate,molecule}

Excluded by decision (2026-10-21): molecules that rearrange when relaxed in isolation (H2
formation / H migration in DFT) and odd-electron (radical) molecules, which spin-paired DFT
treats incorrectly. A run is also dropped (and listed) if its DFT or model molecule, or the
adsorbate in the model-relaxed system, has a different bond graph from the input structure,
or a relaxation did not converge.

Writes eads_small_molecules.csv (variant, label, e_ads, held_out) and prints a summary.
"""
import csv
import os
import sys

import networkx as nx
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_inputs import N_SUB, mol_graph  # noqa: E402

REARRANGED = {"m28_C13H17N5", "m33_C15H16N4", "m40_C18H17N7", "m45_C19H19N5", "m51_C20H16N8",
              "m53_C20H23N3", "m54_C20H25N5", "m56_C22H18N4", "m57_C22H18N4", "m58_C22H18N4"}
RADICALS = {"m05_C7H16N", "m09_C8H18N", "m10_C9H12N", "m14_C10H21N2", "m23_C12H17N2",
            "m44_C18H22N", "m49_C19H26N3"}
VARIANTS = ["m11nod4", "m11_d4trained", "m11nod4_d4"]


def same_bonds(a, ref):
    return nx.is_isomorphic(mol_graph(a), mol_graph(ref), node_match=lambda x, y: x["el"] == y["el"])


def main():
    runs = [r for r in csv.DictReader(open(os.path.join(HERE, "runs.csv")))
            if r["molecule"] not in REARRANGED | RADICALS]
    e_gr = {"DFT": read(os.path.join(HERE, "dft/graphene/graphene.extxyz")).get_potential_energy()}
    for v in VARIANTS:
        e_gr[v] = read(os.path.join(HERE, f"mace/{v}/substrate/graphene.extxyz")).get_potential_energy()

    mol_e, dropped = {}, []
    for m in sorted({r["molecule"] for r in runs}):
        ref = read(os.path.join(HERE, "molecules", f"{m}.xyz"))
        for v, path in [("DFT", f"dft/molecules/{m}.extxyz")] + [(v, f"mace/{v}/molecule/{m}.extxyz") for v in VARIANTS]:
            p = os.path.join(HERE, path)
            if not os.path.exists(p):
                dropped.append((m, v, "molecule missing"))
                continue
            a = read(p)
            if not same_bonds(a, ref):
                dropped.append((m, v, "molecule rearranged"))
                continue
            if v != "DFT" and not a.info.get("converged", False):
                dropped.append((m, v, "molecule not converged"))
                continue
            mol_e[(v, m)] = a.get_potential_energy()

    rows = []
    for r in runs:
        run, m = r["run"], r["molecule"]
        held = "train" not in r["splits"]
        if ("DFT", m) in mol_e:
            rows.append(dict(variant="DFT", label=run, e_ads=float(r["e_system"]) - e_gr["DFT"] - mol_e[("DFT", m)],
                             held_out=held))
        ref_ads = read(os.path.join(HERE, "systems", f"{run}.extxyz"))[N_SUB:]
        for v in VARIANTS:
            p = os.path.join(HERE, f"mace/{v}/system/{run}.extxyz")
            if (v, m) not in mol_e or not os.path.exists(p):
                dropped.append((run, v, "system or molecule missing"))
                continue
            s = read(p)
            if not s.info.get("converged", False):
                dropped.append((run, v, "system not converged"))
                continue
            if not same_bonds(s[N_SUB:], ref_ads):
                dropped.append((run, v, "adsorbate rearranged on surface"))
                continue
            rows.append(dict(variant=v, label=run, e_ads=s.get_potential_energy() - e_gr[v] - mol_e[(v, m)],
                             held_out=held))

    # keep only runs present for DFT and all three models, so every series has the same points
    have = {}
    for x in rows:
        have.setdefault(x["label"], set()).add(x["variant"])
    complete = {k for k, s in have.items() if s == {"DFT", *VARIANTS}}
    rows = [x for x in rows if x["label"] in complete]
    with open(os.path.join(HERE, "eads_small_molecules.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "label", "e_ads", "held_out"])
        w.writeheader()
        w.writerows(rows)
    print(f"runs after exclusions: {len(runs)}; complete for DFT + 3 models: {len(complete)} "
          f"({sum(1 for r in runs if r['run'] in complete and 'train' not in r['splits'])} held out)")
    print(f"dropped ({len(dropped)}):")
    for d in dropped:
        print("  ", *d)
    import numpy as np
    dft = {x["label"]: x["e_ads"] for x in rows if x["variant"] == "DFT"}
    print(f"DFT E_ads range {min(dft.values()):.2f} .. {max(dft.values()):.2f} eV")
    for v in VARIANTS:
        err = np.array([x["e_ads"] - dft[x["label"]] for x in rows if x["variant"] == v])
        print(f"  {v:14s} MAE {np.abs(err).mean():.3f} eV, mean signed {err.mean():+.3f} eV")


if __name__ == "__main__":
    main()
