"""
Candidate adsorption geometries where dispersion beyond MACE's 10 A receptive field should matter,
i.e. where the post-hoc D4 model could beat the learned-dispersion model. All C/H/N on graphene.

  A  multilayer: a converged small-molecule end point (../small_molecules/systems) on 1, 2 and 3
     AB-stacked graphene layers (3.35 A spacing); deeper layers act from > 10 A
  B  second layer: the same molecule pi-stacked (3.4 A, shifted by half a C-C bond) on top of an
     identical flat-lying molecule; E_ads of the top molecule on (graphene + bottom molecule)
  C  upright n-alkylamines C8/C12/C16 (NH2 down, all-trans chain along z) on monolayer graphene

Screening (no DFT): the part of E_ads coming from beyond 10 A, estimated with D3(BJ, PBE,
torch-dftd) as  tail = E_ads^D3(cutoff 50 A) - E_ads^D3(cutoff 10 A), slabs periodic in x,y only,
isolated molecule non-periodic. D3 stands in for D4 here (similar long-range C6 tails); torch-dftd
also cuts the coordination number at 10 A, a meV-level effect for screening.

Writes geometries/<name>_{system,substrate,molecule}.extxyz (graphene, and in B the bottom molecule,
fixed) and screening.csv.  Run in mace_env_cpu_disp (module load gpaw/25.7.0 first).
"""
import csv
import os

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

HERE = os.path.dirname(os.path.abspath(__file__))
SM = os.path.join(os.path.dirname(HERE), "small_molecules")
BASES = [("r00_C76H6N2", "small"), ("r57_C86H14N2", "medium"), ("r100_C93H26N2", "large")]
N_G = 72
CELL_Z = 45.0
LAYER = 3.35


def graphene_and_mol(run):
    s = read(os.path.join(SM, "systems", f"{run}.extxyz"))
    s.set_constraint()
    g, m = s[:N_G], s[N_G:]
    cell = s.cell.copy()
    cell[2] = [0, 0, CELL_Z]
    for a in (g, m):
        a.set_cell(cell)
        a.pbc = (True, True, False)
    return g, m


def layers(g, n):
    a1, a2 = g.cell[0] / 6, g.cell[1] / 6           # 6x6 supercell of the 2.46 A lattice
    ab = (a1 + 2 * a2) / 3                          # in-plane C-C bond vector -> Bernal stacking
    out = g.copy()
    for k in range(1, n):
        lay = g.copy()
        lay.translate(-LAYER * k * np.array([0, 0, 1]) + (ab if k % 2 else 0))
        out += lay
    return out


def alkylamine(n, cell, z0):
    """all-trans C_nH_(2n+1)NH2 along +z, N lowest, N at height z0"""
    cc, half = 1.53, np.radians(112) / 2
    dz, dx = cc * np.sin(half), cc * np.cos(half) / 2
    pos, sym = [], []
    backbone = [np.array([dx * (-1) ** k, 0.0, z0 + 1.47 * np.sin(half) + dz * k]) for k in range(n)]
    pos.append([-dx, 0, z0]); sym.append("N")
    for s_, h in [(1, 1), (-1, 1)]:
        pos.append([-dx - 0.55, s_ * 0.82, z0 - 0.35]); sym.append("H")
    for k, c in enumerate(backbone):
        pos.append(c); sym.append("C")
        side = -np.sign(c[0]) or 1
        nh = 3 if k == n - 1 else 2
        for j in range(nh):
            if j < 2:
                pos.append(c + [0.51 * side, (-1) ** j * 0.89, 0.0]); sym.append("H")
            else:
                pos.append(c + [0.0, 0.0, 1.09]); sym.append("H")
    a = Atoms(sym, positions=pos, cell=cell, pbc=(True, True, False))
    a.translate([cell[0][0] / 2 + cell[1][0] / 2, cell[1][1] / 2, 0])
    return a


def d3(atoms, cut):
    a = atoms.copy()
    a.set_constraint()
    a.calc = TorchDFTD3Calculator(xc="pbe", damping="bj", cutoff=cut, device="cpu", dtype=torch.float64)
    return a.get_potential_energy()


def save(name, sysa, sub, mol, n_fixed):
    d = os.path.join(HERE, "geometries")
    os.makedirs(d, exist_ok=True)
    sysa = sysa.copy()
    sysa.set_constraint(FixAtoms(indices=range(n_fixed)))
    sub = sub.copy()
    sub.set_constraint(FixAtoms(indices=range(len(sub))))
    iso = mol.copy()
    iso.set_cell([0, 0, 0])
    iso.pbc = False
    for tag, a in [("system", sysa), ("substrate", sub), ("molecule", iso)]:
        write(os.path.join(d, f"{name}_{tag}.extxyz"), a, format="extxyz")
    return sysa, sub, iso


def main():
    cands = []
    for run, size in BASES:
        g, m = graphene_and_mol(run)
        for n in (1, 2, 3):
            sub = layers(g, n)
            cands.append((f"A_{size}_{n}layer", "multilayer", sub + m, sub, m, len(sub)))
        top = m.copy()
        top.translate([0.71, 0.0, 3.4])                # pi-stack, half-bond lateral shift
        sub = g + m
        cands.append((f"B_{size}_stacked", "second layer", sub + top, sub, top, len(sub)))
    g, _ = graphene_and_mol(BASES[0][0])
    for n in (8, 12, 16):
        mol = alkylamine(n, g.cell, g.positions[:, 2].max() + 3.0)
        cands.append((f"C_upright_C{n}", "upright", g + mol, g, mol, N_G))

    rows = []
    for name, kind, sysa, sub, mol, nfix in cands:
        sysa, sub, iso = save(name, sysa, sub, mol, nfix)
        e = {cut: d3(sysa, cut) - d3(sub, cut) - d3(iso, cut) for cut in (10.0, 50.0)}
        tail = e[50.0] - e[10.0]
        far = np.max(sysa.positions[nfix:, 2]) - sub.positions[:, 2].min()
        rows.append(dict(name=name, kind=kind, atoms=len(sysa), mol_atoms=len(iso),
                         eads_d3_full=round(e[50.0], 4), eads_d3_10A=round(e[10.0], 4),
                         tail_beyond_10A=round(tail, 4), tail_pct=round(100 * tail / e[50.0], 1),
                         height_span=round(far, 1)))
        print(f"{name:20s} {kind:13s} {len(sysa):4d} atoms | D3 E_ads full {e[50.0]:+.3f} eV, "
              f"beyond 10 A {1000 * tail:+6.0f} meV ({100 * tail / e[50.0]:4.1f}%)")
    with open(os.path.join(HERE, "screening.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
