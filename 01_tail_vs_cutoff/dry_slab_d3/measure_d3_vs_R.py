#!/usr/bin/env python3
"""
Missing D3 energy vs cutoff radius R, whole system, one global cutoff per call.

Replaces the per-atom sphere ("ball") method of
../all_atoms_vs_R/frame_1520/2x2_no_water/measure_d4_tail_vs_R_2x2_no_water.py
for the whole-system curve. For the two-body term the two are the same quantity:
averaging the sphere energies E_i(R) = 1/2 sum_{j: r_ij<R} e_ij over all atoms gives
E_global(R)/N, which is what a single dftd3 call with `-cutoff R` returns.

Method (Grimme's dftd3 binary, `-func pbe -bj -pbc`)
  dftd3 has two cutoffs: `-cutoff` (two-body pairs) and `-cnthr` (coordination numbers
  AND the ATM triplets). They must be varied separately, because the D3 CN counting
  function does not decay to zero: every distant atom adds ~1e-7 to CN, so CN keeps
  growing with `cnthr` (~R^2 atoms). Tying cnthr to R (as ASE forces, cnthr <= cutoff)
  leaks a CN artifact of ~0.02 meV/atom at R = 10 Ang into the two-body curve, the same
  size as the 6x6 tail itself. Measured on the dry 2x2, 2026-10-02.

  E2(R)       two-body energy, `-cutoff R -cnthr 40 bohr` (CN fixed at the D3 default;
              dftd3 accepts cnthr > cutoff, only ASE forbids it)
  E3(R)       ATM energy with triplet cutoff R = E(-abc) - E(no abc), both at
              `-cutoff R -cnthr R` (R < 21.17 Ang); constant for R >= 21.17 Ang. The ATM
              energy depends on cnthr only, not on the pair cutoff (checked: identical to
              1e-6 meV/atom). The CN shift inside E3 is second order (~1e-3 meV/atom).
  references  E2 at 95 bohr (50.3 Ang, dftd3 default), E3 at 40 bohr (21.17 Ang, default)
  missing(R)  two-body:   (E2_ref - E2(R)) / N                  [meV/atom], -> 0
              2-body+ATM: missing_2body + (E3_ref - E3(R)) / N

Notes
  * Needs `module load gpaw/25.7.0` (ships the dftd3 binary; ~/.local python has none).
  * dftd3 is called directly, not through ASE's DFTD3 calculator, because of two ASE
    bugs: (1) grad=False on a periodic cell crashes reading `dftd3_cellgradient`, which
    forces the O(N^2) gradient on; (2) for large cells (6x6, 9144 atoms) the kcal value
    runs into the colon ("Edisp /kcal,au,eV:-16827.2150 -26.81587117 ****") and ASE reads
    the overflowed eV field. Here the Hartree value is parsed instead. Verified to
    reproduce the ASE-based 2x2 energies.
  * dftd3 (Fortran) rejects long file paths, so each call runs in its own directory.
  * dftd3 is single-threaded; R values run in parallel, one process each.
  * Meaningful R is limited to ~half the cell's perpendicular width (periodic images);
    the script prints that limit and shades R beyond it.

Usage
    python3 measure_d3_vs_R.py --workers 16
    python3 measure_d3_vs_R.py --radii 10 15 --workers 4      # quick test
    python3 measure_d3_vs_R.py --structure other.traj --tag 4x4_no_water
"""

import os
import time
import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D4_DIR = os.path.join(HERE, "..", "all_atoms_vs_R", "frame_1520", "2x2_no_water")
DEFAULT_STRUCT = os.path.join(HERE, "..", "all_atoms_vs_R", "frame_1520",
                              "frame_1520_2x2_no_water.traj")
D4_NPZ = os.path.join(D4_DIR, "d4_tail_vs_R_2x2_no_water.npz")
D3_2X2_NPZ = os.path.join(HERE, "2x2_no_water", "d3_vs_R_2x2_no_water.npz")

from ase.units import Bohr as BOHR, Hartree as HARTREE
R_REF = 95 * BOHR                 # dftd3 default two-body cutoff, 50.27 Ang
CNTHR_DEFAULT = 40 * BOHR         # dftd3 default CN / three-body cutoff, 21.17 Ang
RECEPTIVE_FIELD = 10.0            # MACE production receptive field
DEFAULT_RADII = np.arange(10.0, 17.0 + 1e-9, 0.5)


def half_width(cell):
    """Half the smallest perpendicular width of the cell's in-plane (a, b) faces."""
    a, b, c = np.asarray(cell)
    vol = abs(np.dot(a, np.cross(b, c)))
    heights = [vol / np.linalg.norm(np.cross(b, c)),   # perpendicular to the b-c face
               vol / np.linalg.norm(np.cross(a, c))]   # perpendicular to the a-c face
    return 0.5 * min(heights)


def parse_edisp_hartree(text):
    """Dispersion energy in Hartree from dftd3 stdout (the 'au' column of the Edisp line)."""
    for line in text.splitlines():
        if line.lstrip().startswith("Edisp"):
            head, rest = line.split(":", 1)
            cols = head.split("/")[1].strip().split(",")       # ['kcal', 'au', 'eV']
            return float(rest.split()[cols.index("au")])
    raise RuntimeError("no Edisp line in dftd3 output")


def d3_energy(args):
    """One dftd3 call -> (key, E in eV, wall s). Runs in its own directory."""
    key, struct, cut, cn, abc, workdir = args
    from ase.io import read
    from ase.io.vasp import write_vasp

    atoms = read(struct)
    assert all(atoms.pbc), "dftd3 -pbc needs a 3D-periodic cell"
    atoms.set_constraint()                       # no 'Selective dynamics' block
    rundir = os.path.join(workdir, f"cut{cut:07.3f}_cn{cn:07.3f}_abc{int(abc)}")
    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)
    write_vasp("d3.POSCAR", atoms, sort=True)    # sorted, as ASE does (<=10 atom types)
    argv = ["dftd3", "d3.POSCAR", "-func", "pbe", "-pbc",
            "-cnthr", str(cn / BOHR), "-cutoff", str(cut / BOHR), "-bj"]
    if abc:
        argv.append("-abc")
    t = time.time()
    res = subprocess.run(argv, capture_output=True, text=True)
    with open("d3.out", "w") as fh:
        fh.write(" ".join(argv) + "\n" + res.stdout + res.stderr)
    if res.returncode != 0 or "normal termination" not in res.stdout:
        raise RuntimeError(f"dftd3 failed for {key}: see {rundir}/d3.out")
    e = parse_edisp_hartree(res.stdout) * HARTREE
    return key, e, time.time() - t


def make_plot(radii, miss2, miss3, r_limit, tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(radii, miss3, "-o", ms=4, lw=2.4, color="#111111", label="D3(BJ) 2-body + ATM")
    ax.plot(radii, miss2, "-s", ms=4, lw=1.8, color="#0072B2", label="D3(BJ) 2-body only")

    # 1/R^4 guide anchored at the first point of the 2-body curve
    rr = np.linspace(radii[0], radii[-1], 100)
    ax.plot(rr, miss2[0] * (radii[0] / rr) ** 4, ":", lw=1.4, color="#0072B2",
            label=r"$1/R^4$ from 2-body at %.0f $\AA$" % radii[0])

    if os.path.exists(D4_NPZ) and tag == "2x2_no_water":
        d = np.load(D4_NPZ, allow_pickle=True)
        m = d["radii"] <= radii[-1] + 1e-6
        ax.plot(d["radii"][m], d["mean_sys"][m], "--^", ms=4, lw=1.6, color="#D55E00",
                label="D4 sphere method (earlier run)")
    if tag != "2x2_no_water" and os.path.exists(D3_2X2_NPZ):
        d = np.load(D3_2X2_NPZ)
        ax.plot(d["radii"], d["miss3"], "--", lw=1.2, color="#111111", alpha=0.5,
                label="2x2: 2-body + ATM")
        ax.plot(d["radii"], d["miss2"], "--", lw=1.2, color="#0072B2", alpha=0.5,
                label="2x2: 2-body only")

    if r_limit < radii[-1]:
        ax.axvspan(r_limit, radii[-1] + 0.3, color="#999999", alpha=0.15, lw=0)
        ax.text(r_limit + 0.1, 0.03, "periodic images", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#444444")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    ax.axvline(RECEPTIVE_FIELD, color="#D55E00", lw=1.2, ls="--", alpha=0.5)
    ax.set_xlim(radii[0] - 0.3, radii[-1] + 0.3)
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_ylabel(r"missing dispersion $(E_{\rm ref}-E(R))/N$  (meV/atom)")
    ax.set_title(f"D3 energy beyond a global cutoff $R$ ({tag}, ref = {R_REF:.1f} $\\AA$)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default=DEFAULT_STRUCT)
    ap.add_argument("--tag", default="2x2_no_water")
    ap.add_argument("--radii", type=float, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--keep-workdirs", action="store_true")
    ap.add_argument("--ref-check", type=float, default=None,
                    help="extra cutoff (Ang) > 50.27 to confirm the reference is converged")
    args = ap.parse_args()

    if shutil.which("dftd3") is None:
        raise SystemExit("dftd3 binary not found: run `module load gpaw/25.7.0` first")

    from ase.io import read
    atoms = read(args.structure)
    n = len(atoms)
    radii = np.array(sorted(args.radii)) if args.radii else DEFAULT_RADII
    r_limit = half_width(atoms.cell)
    outdir = os.path.join(HERE, args.tag)
    workdir = os.path.join(outdir, "dftd3_runs")
    os.makedirs(workdir, exist_ok=True)

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("=" * 78)
    log("MISSING D3 ENERGY vs GLOBAL CUTOFF R -- PBE-D3(BJ), whole system")
    log("=" * 78)
    log(f"  structure : {os.path.abspath(args.structure)}")
    log(f"  atoms     : {n} ({atoms.get_chemical_formula()})")
    log(f"  cell      : {np.round(atoms.cell.cellpar(), 3)}")
    log(f"  meaningful R <= half perpendicular width = {r_limit:.2f} A")
    log(f"  radii     : {', '.join(f'{r:g}' for r in radii)}  + ref {R_REF:.2f} A")
    log(f"  workers   : {args.workers}")

    S = args.structure
    two_body_R = list(radii) + [R_REF] + ([args.ref_check] if args.ref_check else [])
    atm_R = sorted({min(float(R), CNTHR_DEFAULT) for R in radii} | {CNTHR_DEFAULT})
    tasks = [(("2b", round(float(R), 6)), S, float(R), CNTHR_DEFAULT, False, workdir)
             for R in two_body_R]
    for R in atm_R:
        tasks += [(("atm+", round(R, 6)), S, R, R, True, workdir),
                  (("atm-", round(R, 6)), S, R, R, False, workdir)]
    tasks.sort(key=lambda t: (t[0][0] != "atm+", -t[2]))      # slow ATM + big R first
    log(f"  ATM cutoffs: {', '.join(f'{r:g}' for r in atm_R)}")
    E = {}
    t0 = time.time()
    log("\n--- dftd3 calls (cut = pair cutoff, cn = cnthr) ---")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(d3_energy, t): t for t in tasks}
        for f in as_completed(futs):
            k, e, dt = f.result()
            t = futs[f]
            E[k] = e
            log(f"  {k[0]:4s} cut={t[2]:7.3f} cn={t[3]:7.3f} abc={int(t[4])}  "
                f"E={e:14.6f} eV  ({dt:7.1f} s)")
    log(f"  all {len(tasks)} calls done in {time.time() - t0:.0f} s wall")

    e2_of = lambda R: E[("2b", round(float(R), 6))]
    e3_of = lambda R: (E[("atm+", round(min(float(R), CNTHR_DEFAULT), 6))]
                       - E[("atm-", round(min(float(R), CNTHR_DEFAULT), 6))])
    e2 = np.array([e2_of(R) for R in radii])
    e3atm = np.array([e3_of(R) for R in radii])
    ref2, ref3atm = e2_of(R_REF), e3_of(CNTHR_DEFAULT)
    miss2 = (ref2 - e2) / n * 1000.0
    miss3 = miss2 + (ref3atm - e3atm) / n * 1000.0
    e3, ref3 = e2 + e3atm, ref2 + ref3atm

    log(f"\n  reference E(2-body)     = {ref2:.6f} eV ({1000 * ref2 / n:.4f} meV/atom)")
    log(f"  reference ATM energy    = {ref3atm:.6f} eV ({1000 * ref3atm / n:+.4f} meV/atom, "
        f"triplets to {CNTHR_DEFAULT:.2f} A)")
    log(f"  reference E(2-body+ATM) = {ref3:.6f} eV ({1000 * ref3 / n:.4f} meV/atom)")
    if args.ref_check:
        dref = (e2_of(args.ref_check) - ref2) / n * 1000.0
        log(f"  reference check 2-body: E({args.ref_check:g} A) - E({R_REF:.2f} A) = "
            f"{dref:+.6f} meV/atom")
    log("\n--- missing energy vs R ---")
    log(f"  {'R (A)':>7s} {'2b meV/at':>10s} {'2b+ATM':>10s} {'2b meV/cell':>12s} "
        f"{'2b+ATM cell':>12s} {'1/R^4 pred':>11s}  note")
    for R, m2, m3 in zip(radii, miss2, miss3):
        pred = miss2[0] * (radii[0] / R) ** 4
        note = "beyond half-cell" if R > r_limit + 1e-6 else ""
        log(f"  {R:7.2f} {m2:10.4f} {m3:10.4f} {m2 * n:12.2f} {m3 * n:12.2f} {pred:11.4f}  {note}")

    log("\n--- checks ---")
    mono = lambda m: bool(np.all(np.diff(np.abs(m)) <= 1e-9))
    log(f"  monotone decay 2-body : {'PASS' if mono(miss2) else 'FAIL'}")
    log(f"  monotone decay 2b+ATM : {'PASS' if mono(miss3) else 'FAIL'}")
    m = radii <= r_limit + 1e-6
    if m.sum() >= 3:
        slope = np.polyfit(np.log(radii[m]), np.log(-miss2[m]), 1)[0]
        log(f"  log-log slope of 2-body over {radii[m][0]:g}-{radii[m][-1]:g} A: "
            f"{slope:.2f}  (2D sheet ~ -4, 3D bulk ~ -3)")

    np.savez(os.path.join(outdir, f"d3_vs_R_{args.tag}.npz"), radii=radii, e2=e2, e3=e3,
             ref2=ref2, ref3=ref3, miss2=miss2, miss3=miss3, n=n, r_limit=r_limit,
             r_ref=R_REF, structure=os.path.abspath(args.structure))
    np.savetxt(os.path.join(outdir, f"d3_vs_R_{args.tag}.csv"),
               np.column_stack([radii, e2, e3, miss2, miss3]), delimiter=",", fmt="%.8f",
               header="R_A,E_2body_eV,E_2body_plus_ATM_eV,missing_2body_meV_per_atom,"
                      "missing_2body_ATM_meV_per_atom", comments="")
    png = os.path.join(outdir, f"d3_vs_R_{args.tag}.png")
    make_plot(radii, miss2, miss3, r_limit, args.tag, png)
    log(f"\n  wrote {png}")
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if not args.keep_workdirs:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
