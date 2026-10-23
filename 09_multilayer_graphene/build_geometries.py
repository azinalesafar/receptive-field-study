"""
Bilayer and trilayer versions of 4 DFT-relaxed candidate cells (see README.md).
Layers are added below the original graphene (fixed), AB(A) stacked via the in-plane C-C bond
vector, 3.35 A spacing, same cell (24 A, pbc T T F) as the original DFT. Writes
geometries/<cell>_<n>L_{system,substrate}.extxyz for n = 1, 2, 3 (1L = original geometry).
"""
import os

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write

D = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT"
CELLS = ["m-8_n0_m-6_n-8", "m0_n8_m8_n6", "m-8_n-1_m-5_n-8", "m-8_n-6_m-5_n3"]
N_ADS, LAYER = 126, 3.35
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometries")

for c in CELLS:
    s = read(f"{D}/unitcell_candidates/DFT_relaxed/{c}/{c}_system_relaxed.xyz")
    n = len(s) - N_ADS
    g, m = s[:n].copy(), s[n:].copy()
    d = g.get_distances(0, range(1, n), mic=True, vector=True)
    bond = d[np.argmin(np.linalg.norm(d, axis=1))]
    bond[2] = 0.0
    for nl in (1, 2, 3):
        sub = g.copy()
        for k in range(1, nl):
            lay = g.copy()
            lay.translate(np.array([0, 0, -LAYER * k]) + (bond if k % 2 else 0))
            sub += lay
        sysa = sub + m
        assert sysa.positions[:, 2].min() > 3.0, "layer too close to the box edge"
        sysa.set_constraint(FixAtoms(indices=range(len(sub))))
        sub.set_constraint(FixAtoms(indices=range(len(sub))))
        for tag, a in (("system", sysa), ("substrate", sub)):
            a.calc = None
            write(f"{OUT}/{c}_{nl}L_{tag}.extxyz", a, format="extxyz")
        print(f"{c} {nl}L: system {len(sysa)} atoms, lowest z {sysa.positions[:, 2].min():.2f} A, box {s.cell[2, 2]:.0f} A")
