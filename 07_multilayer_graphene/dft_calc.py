"""
GPAW + D4 with the exact settings of model_01's data and the candidate-cell DFT
(GPAW PW(500), PBE, energy conv 1e-6, DFTD4(method='PBE') via SumCalculator; substrate fixed).

  python dft_calc.py sp    <in> <outdir> [kx ky kz]   single point (default k 2 2 1)
  python dft_calc.py relax <in> <outdir>              isolated-molecule relaxation, set up like
        DFT/isolated_adsorbate/PreconLBFGSAdsorption-relax.py: center(vacuum=8), pbc True, k=(1,1,1),
        PreconLBFGS(Exp, maxstep 0.02), fmax 0.01. If PreconLBFGS's line search fails, FIRE takes
        over; a rerun resumes from the last frame of <base>.traj.
Writes <outdir>/<base>.extxyz (energy, forces; info: e_pbe = GPAW part, e_d4 = D4 part) and the GPAW log.
"""
import os
import sys

from ase.calculators.mixing import SumCalculator
from ase.io import read, write
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Exp
from dftd4.ase import DFTD4
from gpaw import GPAW, PW

mode, inp, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
base = os.path.splitext(os.path.basename(inp))[0]
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, f"{base}.extxyz")
if os.path.exists(out):
    print(f"{out} exists, nothing to do")
    sys.exit(0)

atoms = read(inp)
traj = os.path.join(outdir, f"{base}.traj")
resume = False
if mode == "relax":
    if os.path.exists(traj) and os.path.getsize(traj) > 0:
        # an earlier attempt stopped (e.g. PreconLBFGS line-search failure): continue from its
        # last geometry (already centred, same box) instead of starting the DFT relax over
        atoms = read(traj, -1)
        resume = True
    else:
        atoms.center(vacuum=8.0)
        atoms.set_pbc(True)
    kpts = (1, 1, 1)
else:
    kpts = tuple(int(k) for k in sys.argv[4:7]) if len(sys.argv) > 4 else (2, 2, 1)

gpaw = GPAW(mode=PW(500), xc="PBE", kpts=kpts, convergence={"energy": 1e-6},
            txt=os.path.join(outdir, f"{base}.log"))
d4 = DFTD4(method="PBE")
atoms.calc = SumCalculator([gpaw, d4])
if mode == "relax":
    from ase.optimize import FIRE
    fire_kw = dict(maxstep=0.05, trajectory=os.path.join(outdir, f"{base}_fire.traj"),
                   logfile=os.path.join(outdir, f"{base}_fire_opt.log"))
    if resume:
        print(f"resuming {base} from the last frame of {traj} with FIRE")
        FIRE(atoms, **fire_kw).run(fmax=0.01)
        atoms.info["optimizer"] = "PreconLBFGS->FIRE (resumed)"
    else:
        try:
            PreconLBFGS(atoms, precon=Exp(), maxstep=0.02, use_armijo=False, trajectory=traj,
                        logfile=os.path.join(outdir, f"{base}_opt.log")).run(fmax=0.01)
        except RuntimeError as err:   # Wolff line search gave up; the minimum is optimizer-independent
            print(f"PreconLBFGS stopped ({err}); continuing with FIRE")
            FIRE(atoms, **fire_kw).run(fmax=0.01)
            atoms.info["optimizer"] = "PreconLBFGS->FIRE"
e = atoms.get_potential_energy()
atoms.get_forces()
atoms.info.update(e_pbe=gpaw.get_potential_energy(atoms), e_d4=d4.get_potential_energy(atoms), kpts=str(kpts))
write(out, atoms, format="extxyz")
print(f"{base}: E = {e:.6f} eV (PBE {atoms.info['e_pbe']:.6f}, D4 {atoms.info['e_d4']:.6f})")
