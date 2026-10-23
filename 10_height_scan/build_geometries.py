"""
Height scan: the DFT-relaxed molecule of one candidate cell is lifted rigidly along z by dh above
its relaxed position on single-layer graphene (see README.md). The box is enlarged from 24 to 37 A
(graphene stays at the same z) so the largest lift still has >= 10 A vacuum above the molecule.
Writes geometries/scan_dh<dh>.extxyz, plus substrate.extxyz and molecule.extxyz (molecule alone,
dh = 0, same cell and pbc, so in-plane image interactions cancel in E_int).
"""
import os

from ase.constraints import FixAtoms
from ase.io import read, write

D = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT"
CELL = "m-8_n0_m-6_n-8"
N_ADS, BOX_Z = 126, 37.0
SHIFTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometries")

s = read(f"{D}/unitcell_candidates/DFT_relaxed/{CELL}/{CELL}_system_relaxed.xyz")
s.calc = None
c = s.cell.copy()
c[2, 2] = BOX_Z
s.set_cell(c, scale_atoms=False)
n = len(s) - N_ADS
g, m = s[:n].copy(), s[n:].copy()
g.set_constraint(FixAtoms(indices=range(n)))
write(f"{OUT}/substrate.extxyz", g, format="extxyz")
write(f"{OUT}/molecule.extxyz", m, format="extxyz")

for dh in SHIFTS:
    mm = m.copy()
    mm.translate([0, 0, dh])
    a = g + mm
    a.set_constraint(FixAtoms(indices=range(n)))
    top = a.positions[:, 2].max()
    assert BOX_Z - top >= 10.0, "not enough vacuum above the molecule"
    gap = mm.positions[:, 2].min() - g.positions[:, 2].max()
    write(f"{OUT}/scan_dh{dh:.1f}.extxyz", a, format="extxyz")
    print(f"dh {dh:4.1f} A: min molecule-graphene z gap {gap:5.2f} A, vacuum above {BOX_Z - top:5.2f} A")
