"""
Relax (or single-point) one structure with a MACE model, optionally adding a
dispersion correction at inference time:  E = E_MACE + E_disp  (ASE SumCalculator),
used for every force call, i.e. the dispersion acts DURING the relaxation.
This is the "post-hoc vdW" scheme of MAD-SURF SI Fig. S18 (they used torch-dftd D3).

Same optimizer settings as MACE_model01/mace-single-structure-relax.py:
PreconLBFGS(Exp), maxstep 0.02, fmax 0.01 eV/A, 500 steps, float64.
FixAtoms constraints carried by the input file are kept (substrate fixed).

Dispersion options
  none : MACE only
  d4   : dftd4.ase.DFTD4(method="PBE"), dftd4 defaults -- identical to the
         calculator that generated the PBE+D4 training labels
  d3   : torch-dftd D3(BJ), PBE, two-body.  --disp-cutoff sets the pair cutoff
         (default 95 bohr = 50.3 A).  NOTE: torch-dftd clamps the CN cutoff to the
         pair cutoff when cutoff < 40 bohr (21.17 A), so at 15 A the CN is also
         cut at 15 A.  Size of this effect is checked separately with the dftd3
         binary (see check_d3_cn_coupling.py).
D4 has no cutoff option in dftd4 3.7 (and tad-dftd4 has no PBC), so no D4 at 15 A.

Needs the x86 env (dftd4 lives only in the x86 gpaw module):
  module load gpaw/25.7.0; source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
"""
import argparse
import os
import time

import torch
from ase.calculators.mixing import SumCalculator
from ase.io import read, write
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Exp

# The .model files were saved on GPU: e3nn's TorchScript blobs inside them are
# reloaded with torch.jit.load, which ignores torch.load's map_location and
# tries to allocate on CUDA.  Default it to CPU.
_jit_load = torch.jit.load


def _jit_load_cpu(f, *args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _jit_load(f, *args, **kwargs)


torch.jit.load = _jit_load_cpu

from mace.calculators import MACECalculator  # noqa: E402


def make_calc(model, dispersion, disp_cutoff):
    mace = MACECalculator(model_paths=[model], device="cpu", default_dtype="float64")
    if dispersion == "none":
        return mace
    if dispersion == "d4":
        if disp_cutoff is not None:
            raise SystemExit("dftd4 has no cutoff option; d4 runs at full range only")
        from dftd4.ase import DFTD4
        return SumCalculator([mace, DFTD4(method="PBE")])
    if dispersion == "d3":
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        kw = dict(xc="pbe", damping="bj", abc=False, device="cpu", dtype=torch.float64)
        if disp_cutoff is not None:
            kw["cutoff"] = disp_cutoff
        return SumCalculator([mace, TorchDFTD3Calculator(**kw)])
    raise ValueError(dispersion)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("input_file")
    p.add_argument("--name", required=True, help="output basename")
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dispersion", choices=["none", "d3", "d4"], default="none")
    p.add_argument("--disp-cutoff", type=float, default=None, help="Angstrom (d3 only)")
    p.add_argument("--mode", choices=["relax", "sp"], default="relax")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--optimizer", choices=["precon", "fire"], default="precon",
                   help="fire = skip PreconLBFGS (it hangs / produces NaN on a few geometries)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.name}.extxyz")
    if os.path.exists(out):
        print(f"{out} exists, nothing to do")
        return

    atoms = read(args.input_file, index=0)
    if not atoms.pbc.any() and atoms.cell.volume < 1e-6:
        # PreconLBFGS needs a cell to estimate neighbour distances; a vacuum box with
        # pbc off changes nothing physically (MACE/D3/D4 see no images).
        atoms.center(vacuum=15.0)
    print(f"{args.name}: {len(atoms)} atoms, pbc {atoms.pbc}, constraints {atoms.constraints}")
    print(f"model {args.model}\ndispersion {args.dispersion} cutoff {args.disp_cutoff} mode {args.mode}")
    atoms.calc = make_calc(args.model, args.dispersion, args.disp_cutoff)

    t0 = time.time()
    converged, nsteps = True, 0
    if args.mode == "relax" and args.optimizer == "fire":
        from ase.optimize import FIRE
        fire = FIRE(atoms, maxstep=0.05, logfile=os.path.join(args.outdir, f"{args.name}_fire.log"),
                    trajectory=os.path.join(args.outdir, f"{args.name}_fire.traj"))
        converged = bool(fire.run(fmax=args.fmax, steps=max(args.steps, 1000)))
        nsteps = fire.get_number_of_steps()
        atoms.info["optimizer"] = "FIRE"
    elif args.mode == "relax":
        opt = PreconLBFGS(atoms, precon=Exp(), maxstep=0.02, use_armijo=False,
                          logfile=os.path.join(args.outdir, f"{args.name}.log"),
                          trajectory=os.path.join(args.outdir, f"{args.name}.traj"))
        try:
            converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
            nsteps = opt.get_number_of_steps()
        except RuntimeError as err:
            # PreconLBFGS's Wolff line search can give up on flat surfaces (near-converged
            # starts, floppy molecules). Continue from the current geometry with FIRE; the
            # minimum does not depend on the optimizer. Recorded in info["optimizer"].
            nsteps = opt.get_number_of_steps()
            print(f"PreconLBFGS stopped after {nsteps} steps ({err}); continuing with FIRE")
            from ase.optimize import FIRE
            fire = FIRE(atoms, maxstep=0.05, logfile=os.path.join(args.outdir, f"{args.name}_fire.log"),
                        trajectory=os.path.join(args.outdir, f"{args.name}_fire.traj"))
            converged = bool(fire.run(fmax=args.fmax, steps=max(args.steps - nsteps, 200)))
            nsteps += fire.get_number_of_steps()
            atoms.info["optimizer"] = "PreconLBFGS->FIRE"

    energy = atoms.get_potential_energy()
    fmax = (atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5
    atoms.info.update(model=args.model, dispersion=args.dispersion,
                      disp_cutoff=-1.0 if args.disp_cutoff is None else args.disp_cutoff,
                      mode=args.mode, converged=converged, steps=nsteps, final_fmax=fmax)
    write(out, atoms, format="extxyz")
    print(f"E = {energy:.6f} eV  fmax = {fmax:.4f} eV/A  converged={converged}  steps={nsteps}  "
          f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
