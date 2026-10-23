"""
Turn model_01's DFT training data (60 small C/H/N molecules on a fixed C72 graphene sheet,
SPICE_MACE/model_01/dataset/) into extra adsorption-energy data points.

  1. split the 481 frames into individual relaxation runs: within a run the fixed graphene
     never moves, so (formula, graphene position) identifies a run
  2. per run take the lowest-force frame as the DFT end point (E_system); runs whose best
     frame has max|F| on the adsorbate > FMAX_OK are listed but not used
  3. group runs by molecule (bond graph of the adsorbate, element-labelled isomorphism), so
     each distinct molecule gets ONE isolated-molecule DFT relaxation
  4. write the inputs:
       systems/<run>.extxyz     DFT end point (stored DFT energy/forces), FixAtoms on graphene
       molecules/<mol>.xyz      adsorbate geometry from one end point, start of the isolated relax
       graphene.extxyz          the bare sheet (identical in every frame up to a rigid shift)
       runs.csv                 run id, molecule id, formula, splits, frames, best fmax, E_system
Source files are only read.

Run: module load gpaw/25.7.0; source .../SPICE_MACE/mace_env_cpu_disp/bin/activate; python3 prepare_inputs.py
"""
import csv
import os
from collections import defaultdict

import networkx as nx
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

SRC = "/scratch/johnt447/azin/SPICE_MACE/model_01/dataset"
FILES = {"train": "training_data_combined.xyz", "valid": "validating_data.extxyz",
         "test": "test_data_combined.xyz"}
HERE = os.path.dirname(os.path.abspath(__file__))
N_SUB = 72
FMAX_OK = 0.02      # eV/A on the adsorbate; looser end points are not used


def mol_graph(m):
    nl = NeighborList(natural_cutoffs(m, mult=1.15), self_interaction=False, bothways=True)
    nl.update(m)
    g = nx.Graph()
    for i, s in enumerate(m.get_chemical_symbols()):
        g.add_node(i, el=s)
    for i in range(len(m)):
        for j in nl.get_neighbors(i)[0]:
            if j > i:
                g.add_edge(i, j)
    return g


def main():
    runs = defaultdict(list)          # key -> [(split, atoms)]
    for split, fn in FILES.items():
        for a in read(os.path.join(SRC, fn), ":"):
            key = (a.get_chemical_formula(), tuple(np.round(a.positions[0], 3)))
            runs[key].append((split, a))

    for d in ("systems", "molecules"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    graphs, rows, skipped = [], [], []   # graphs: [(mol_id, graph)]
    for k, (key, frames) in enumerate(sorted(runs.items(), key=lambda kv: kv[0][0])):
        fmax = [np.linalg.norm(a.get_forces()[N_SUB:], axis=1).max() for _, a in frames]
        best = frames[int(np.argmin(fmax))][1]
        run_id = f"r{k:02d}_{key[0]}"
        splits = "+".join(sorted({s for s, _ in frames}))
        if min(fmax) > FMAX_OK:
            skipped.append((run_id, min(fmax)))
            continue
        mol = best[N_SUB:].copy()
        g = mol_graph(mol)
        if nx.number_connected_components(g) != 1:
            skipped.append((run_id, "adsorbate not one molecule"))
            continue
        match = next((mid for mid, h in graphs
                      if nx.is_isomorphic(g, h, node_match=lambda x, y: x["el"] == y["el"])), None)
        if match is None:
            match = f"m{len(graphs):02d}_{mol.get_chemical_formula()}"
            graphs.append((match, g))
            iso = mol.copy()
            iso.set_cell([0, 0, 0])
            iso.pbc = False
            write(os.path.join(HERE, "molecules", f"{match}.xyz"), iso, format="extxyz")
        sysa = best.copy()
        sysa.set_constraint(FixAtoms(indices=range(N_SUB)))
        sysa.calc = SinglePointCalculator(sysa, energy=best.get_potential_energy(), forces=best.get_forces())
        write(os.path.join(HERE, "systems", f"{run_id}.extxyz"), sysa, format="extxyz")
        rows.append(dict(run=run_id, molecule=match, formula=key[0], splits=splits, frames=len(frames),
                         best_fmax=round(min(fmax), 4), e_system=best.get_potential_energy()))

    g0 = rows and read(os.path.join(HERE, "systems", rows[0]["run"] + ".extxyz"))[:N_SUB]
    g0.set_constraint()
    write(os.path.join(HERE, "graphene.extxyz"), g0, format="extxyz")

    with open(os.path.join(HERE, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    held = [r for r in rows if "train" not in r["splits"]]
    print(f"{len(runs)} relaxation runs; usable {len(rows)} (best max|F| <= {FMAX_OK}); "
          f"skipped {len(skipped)}: {skipped}")
    print(f"{len(graphs)} distinct molecules -> isolated-molecule relaxations to run")
    print(f"runs never in model_01's training split: {len(held)}")
    sizes = sorted(len(read(os.path.join(HERE, 'molecules', m + '.xyz'))) for m, _ in graphs)
    print(f"molecule sizes: {sizes}")


if __name__ == "__main__":
    main()
