"""
Charge-transfer check for the height scan (see README.md): the PBE part of the DFT E_int does not
decay (-0.54 eV at 11 A) and the z-dipole grows linearly with height, i.e. ~0.3 e stays transferred.
Recompute one far lift and its references with the scan settings except:

  w001       Fermi-Dirac width 0.01 eV (scan: 0.1 eV), spin-paired
  w001_spin  same, spin-polarized; initial moments 0.5 on the molecule's N atoms to break symmetry

  python dft_ct_check.py <setting> <in.extxyz>   ->  dft_ct_check/<setting>/<base>.extxyz + .log

Otherwise identical to ../multilayer_test/dft_calc.py sp: GPAW PW(500), PBE, k 2x2x1, energy
convergence 1e-6, DFTD4(method='PBE') via SumCalculator. info: e_pbe, e_d4, dipole_z, magmom.
"""
import os
import sys

from ase.calculators.mixing import SumCalculator
from ase.io import read, write
from dftd4.ase import DFTD4
from gpaw import GPAW, PW, FermiDirac

N_GRAPHENE = 128
setting, inp = sys.argv[1], sys.argv[2]
outdir = os.path.join("dft_ct_check", setting)
os.makedirs(outdir, exist_ok=True)
base = os.path.splitext(os.path.basename(inp))[0]
out = os.path.join(outdir, f"{base}.extxyz")
if os.path.exists(out):
    print(f"{out} exists, nothing to do")
    sys.exit(0)

atoms = read(inp)
spin = setting.endswith("_spin")
if spin:
    off = 0 if base == "molecule" else N_GRAPHENE
    mm = [0.5 if (i >= off and s == "N") else 0.0 for i, s in enumerate(atoms.get_chemical_symbols())]
    atoms.set_initial_magnetic_moments(mm)

gpaw = GPAW(mode=PW(500), xc="PBE", kpts=(2, 2, 1), convergence={"energy": 1e-6},
            occupations=FermiDirac(0.01), spinpol=spin, txt=os.path.join(outdir, f"{base}.log"))
d4 = DFTD4(method="PBE")
atoms.calc = SumCalculator([gpaw, d4])
e = atoms.get_potential_energy()
atoms.get_forces()
atoms.info.update(e_pbe=gpaw.get_potential_energy(atoms), e_d4=d4.get_potential_energy(atoms),
                  dipole_z=float(gpaw.get_dipole_moment()[2]),
                  magmom=float(gpaw.get_magnetic_moment()) if spin else 0.0, setting=setting)
write(out, atoms, format="extxyz")
print(f"{setting} {base}: E = {e:.6f} eV (PBE {atoms.info['e_pbe']:.6f}, D4 {atoms.info['e_d4']:.6f}), "
      f"dipole_z {atoms.info['dipole_z']:.3f} e*A, magmom {atoms.info['magmom']:.3f}")
