#!/usr/bin/env python3
"""
How much DFT-D4 dispersion lives beyond MACE's 10 Ang receptive field, for the
real slab/water production system -- and does it matter?

THE DECISION THIS ANSWERS
-------------------------
model10_stagetwo.model was fine-tuned on labels that already contain D4
("dispersion baked into the training data"). The alternative is to strip D4 from
the labels, retrain the whole two-stage lineage, and add D4 back analytically at
inference. That costs weeks of GPU budget, and -- measured 2026-09-17 -- is not
even usable for production MD afterwards: dftd4 on the 2087-atom production
system costs ~10 min per energy+force call, against 8,000,000 MD steps.

The only thing that would justify it is dispersion living beyond the model's
10.0 Ang receptive field (r_max 5.0 x 2 layers, model10_965159.out:63), where no
amount of fitting can represent it. This script measures exactly that.

RETRACTION -- READ THIS BEFORE QUOTING tail3.txt
------------------------------------------------
The earlier measurement in this directory (measure_d4_tail.py -> tail3.txt)
reported 0.03% / 0.0075 meV per atom for bulk water. THAT NUMBER IS WRONG.
Independent re-measurement with the exact pairwise decomposition used here gives
a value 50-100x larger. The old estimator was E(A+B) - E(A) - E(B) on sampled
molecule pairs; beyond ~14 Ang that difference sat at the float64 noise floor and
returned POSITIVE shell energies for a strictly attractive interaction, and the
positive noise then cancelled real attractive contributions. Its own printed
table shows this. Run --bulk-water-check to reproduce the correct number.
tail3.txt must not be cited again.

WHY THIS MEASUREMENT IS EXACT
-----------------------------
Three facts, each re-verified at runtime rather than assumed (see V2, V4, V8):

  1. D4 here is strictly finite-ranged. dftd4 3.7.0 has hard internal real-space
     cutoffs: the two-body term is exactly 0.0 beyond 60 bohr = 31.75 Ang and the
     three-body (ATM) term beyond 40 bohr = 21.17 Ang. So there is nothing to
     extrapolate -- the question is the bounded one "how much D4 lives in the
     annulus 10 < r <= 31.75 Ang", and the old script's r^-6 extrapolation
     estimated a quantity that is identically zero in the label.

  2. get_pairwise_dispersion() returns (N,N) additive and non-additive matrices
     whose sum equals the total energy to machine precision, and which are
     exactly symmetric -- in the PERIODIC case too. So E_i = sum_j M[i,j] is an
     exact per-atom energy with sum_i E_i = E_total. No sampling, no additivity
     assumption, and no double-counting bookkeeping to get wrong.

  3. D4's C6 coefficients depend on EEQ charges and coordination numbers, which
     are formally non-local, so truncating to a ball is not obviously safe. It is:
     across ball radii 5->14 Ang the CN is bit-identical and the polarizability
     converges to ~1e-5 relative by R ~ 9 Ang. V8 re-measures this and prints it.

THE ESTIMATORS
--------------
    E_i^tail(R) = E_i^ref(periodic) - E_i^ball(R)
    dF_i(R)     = F_i^ref(periodic) - F_i^ball(R)

E_i^tail is precisely the site energy a 10 Ang-receptive-field model cannot
construct. dF_i is the right comparator for forces rather than an approximation:
a MACE force F_i = -d(sum_k E_k)/dr_i depends directly on geometry within 10 Ang
of i, which is the ball.

No damping is applied to the ball, and damping would be WRONG here: MACE's radial
basis goes smoothly to zero at r_max, so its effective weight near the cutoff is
LOWER than a hard sphere's. A hard-cutoff dF therefore understates, not inflates,
what is missing. dF is reported as an upper bound and the R-scan settles the
sensitivity empirically.

WHAT IS AND IS NOT IN THE DECISION RULE
---------------------------------------
The MEAN tail per atom is deliberately EXCLUDED. A configuration-independent
per-element offset is absorbed by MACE's E0s and scale-shift during fitting; bare
graphene's mean tail is ~0.8 meV/atom, above the model's own 0.62 meV/atom RMSE,
and is pure constant. What matters is the part that VARIES and the part that is
unlearnable inside 10 Ang -- sigma_E, sigma_irred, the force statistics, and the
registry spread. See print_verdict().

CAVEATS
-------
  - Dispersion only. The electrostatic 1/r tail is the genuinely long-ranged term
    at an interface and nothing here bears on it. This answers whether stripping
    D4 buys anything, not whether 10 Ang is adequate in general.
  - The available trajectories all freeze graphene and adsorbate (FixAtoms,
    md_main_system_fixed_adsorbate_substrate.py:140), so the frame-to-frame
    variation measured is water-only and is a LOWER bound. The registry module
    partly compensates by varying the substrate registry explicitly.
  - get_pairwise_dispersion is a decomposition convention, not a unique physical
    partition. Totals and region sums are convention-independent; the per-atom
    z-profile is not. All three are reported.

USAGE
-----
    module load gpaw/25.7.0
    python3 measure_d4_tail_slab.py --frames 10 --adsorption --save

The script re-execs itself once to fix the OpenMP environment (dftd4 segfaults
otherwise -- and `ulimit -s unlimited` alone does NOT fix it). Nothing outside
this directory is ever written.
"""

import argparse
import glob
import os
import re
import resource
import sys
import time
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# Environment. This must run before dftd4 is imported anywhere, so dftd4 and
# matplotlib are imported lazily inside functions, never at module scope.
# ----------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "/scratch/johnt447/azin"


def ensure_env():
    """Re-exec once with a sane OpenMP environment, then raise the stack limit.

    dftd4's Fortran core overflows the *OpenMP worker thread* stack on systems of
    this size. `ulimit -s` governs the main thread only, so it is not sufficient
    on its own -- measured: a 2087-atom cluster segfaults (exit 139) with the
    stack already unlimited if OMP_NUM_THREADS is left at its default. Both knobs
    are needed. Single-threaded is also dramatically faster here (600 atoms:
    0.88 s at 1 thread vs 24.7 s at 24), because the work per call is small and
    the threading overhead dominates.

    The re-exec matters rather than setting os.environ in place: the OpenMP
    runtime reads these at initialisation, so patching them after the fact in a
    process that has already touched a threaded library is unreliable.
    """
    if os.environ.get("D4TAIL_REEXEC") != "1":
        env = dict(os.environ)
        env.update({
            "D4TAIL_REEXEC": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_STACKSIZE": "1G",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        })
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = 1 << 30
    if soft != resource.RLIM_INFINITY and soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
        except (ValueError, OSError):
            pass
        soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != resource.RLIM_INFINITY and soft < want:
            sys.exit("stack limit too small for dftd4 at these sizes.\n"
                     "Run:  ulimit -s unlimited   and try again.")


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

BOHR = 0.529177210903
HARTREE = 27.211386245988
HA_PER_BOHR_TO_EV_PER_ANG = HARTREE / BOHR

R_MAX_MODEL = 5.0
N_LAYERS = 2
RECEPTIVE_FIELD = R_MAX_MODEL * N_LAYERS      # 10.0 Ang

D4_CUT_TWOBODY = 31.7502                      # 60 bohr, re-measured by V2
D4_CUT_THREEBODY = 21.1668                    # 40 bohr, re-measured by V2

# Decision thresholds, as fractions of the model's own fitting error.
KEEP_FRAC = {"F_rms": 0.15, "F_sys": 0.10, "sigma_E": 0.15, "sigma_irred": 0.30}
RETRAIN_FRAC = {"F_rms": 0.50, "F_sys": 0.50, "sigma_E": 0.50, "sigma_irred": 0.50}
DELTA_ADS_KEEP_MEV = 3.0
DELTA_ADS_RETRAIN_MEV = 7.0

MAIN_SYS = os.path.join(
    REPO, "SPICE_MACE/compare_models/dynamics/main_system")
REFERENCE_STRUCT = os.path.join(MAIN_SYS, "system_with_water.extxyz")
MODEL10_LOG = os.path.join(
    REPO, "SPICE_MACE/model10_but_from_model11_being_foundation/model10_965159.out")
REGRESSION_STRUCT = os.path.join(REPO, "analysis/ads_eng_calc/DFT/cp2k_gpaw_relaxed_final.extxyz")
REGRESSION_PERIODIC_EV = -40.455719
REGRESSION_ISOLATED_EV = -35.906271
WATER_BOX = os.path.join(
    REPO, "datasets/bulk_water/sp_results_ang/frame_0000/frame_0000/frame_0000_sp.extxyz")
ADS_CSV = os.path.join(REPO, "analysis/ads_eng_calc/DFT/adsorption_energies.csv")
SUBSTRATE_GLOB = os.path.join(REPO, "analysis/ads_eng_calc/DFT/substrate/sp_results/*/*_sp.extxyz")
SYSTEM_GLOB = os.path.join(
    REPO, "analysis/ads_eng_calc/DFT/unitcell_candidates/DFT_relaxed/*/*_system_relaxed.xyz")
ISOLATED_ADS = os.path.join(
    REPO, "analysis/ads_eng_calc/DFT/isolated_adsorbate/relaxed_results/isolated_adsorbate/"
          "isolated_adsorbate_system_relaxed.xyz")

ADSORBATE_FORMULA = Counter({"C": 56, "H": 62, "N": 8})
GRAPHENE_MIN_SIZE = 20


# ----------------------------------------------------------------------------
# Read-only contract
# ----------------------------------------------------------------------------

def guard_write(path, force=False):
    """Refuse to write anywhere but this directory, and refuse silent clobber."""
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(THIS_DIR) + os.sep):
        raise RuntimeError(f"refusing to write outside {THIS_DIR}: {real}")
    for forbidden in (".model", ".pt", ".traj"):
        if real.endswith(forbidden):
            raise RuntimeError(f"refusing to write a {forbidden} file: {real}")
    if os.path.exists(real) and not force:
        raise RuntimeError(f"{real} exists; pass --force to overwrite")
    return real


def guard_read(path):
    """Structures and logs are read; training data and models never are."""
    real = os.path.realpath(path)
    if real.endswith(".model") or "/dataset/" in real:
        raise RuntimeError(f"refusing to read model/training data: {real}")
    return real


class Tee:
    """Mirror stdout into the transcript file."""

    def __init__(self, stream, path=None):
        self.stream = stream
        self.fh = open(path, "w") if path else None

    def write(self, s):
        self.stream.write(s)
        if self.fh:
            self.fh.write(s)

    def flush(self):
        self.stream.flush()
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ----------------------------------------------------------------------------
# Region assignment
#
# These three functions are copied verbatim from
#   SPICE_MACE/compare_models/dynamics/main_system/md_main_system_fixed_adsorbate_substrate.py:56-97
# rather than imported, because that module parses argv at import time and would
# hijack this script's CLI. Keep them in sync if the original changes.
# ----------------------------------------------------------------------------

def bonded_components(atoms):
    """Connected components of the covalent bonding graph (bonds found via
    natural cutoffs, periodic images included)."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cutoffs = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    n = len(atoms)
    rows, cols = [], []
    for i in range(n):
        neighbors, _ = nl.get_neighbors(i)
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    n_components, labels = connected_components(adjacency, directed=False)
    return [np.where(labels == c)[0] for c in range(n_components)]


def find_adsorbate_indices(components, symbols, formula=ADSORBATE_FORMULA):
    matches = [idx for idx in components if Counter(symbols[idx]) == formula]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one component matching "
                           f"{dict(formula)}, found {len(matches)}.")
    return matches[0]


def find_graphene_indices(components, symbols, min_size=GRAPHENE_MIN_SIZE):
    matches = [idx for idx in components
               if len(idx) >= min_size and set(symbols[idx]) == {"C"}]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one pure-carbon fragment with >= "
                           f"{min_size} atoms, found {len(matches)}.")
    return matches[0]


REGIONS = ["graphene", "adsorbate", "interfacial_water", "bulk_water", "water_surface"]


def assign_regions(atoms, topology=None):
    """Label every atom by region, and return z measured from the graphene plane.

    The interface is strongly asymmetric -- sheet + adlayer on one side, ~58 Ang
    of water on the other -- so the tail does not cancel and the physics argument
    hinges on the interface-normal profile. Hence regions rather than a single
    number.

    `topology` supplies the structure used for the *bonded-component* step only;
    z and the water regions still come from `atoms`. Pass the trajectory's start
    structure. On a hot MD frame water H-bonds to within ~1.9-2.2 Ang of the
    adsorbate, natural_cutoffs(mult=1.2) then calls that a covalent bond, and the
    adsorbate's component comes back as C56H68N8O3 -- which matches no formula and
    killed job 1389393 on 9 of 10 frames. Graphene (0-127) and the adsorbate
    (128-253) are under FixAtoms and do not move, so the start structure gives the
    same indices without the contamination.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    comps = bonded_components(topology if topology is not None else atoms)
    ads = find_adsorbate_indices(comps, symbols)
    gra = find_graphene_indices(comps, symbols)

    # The premise above -- that these indices are transferable between frames --
    # holds only because both fragments are frozen. Check it rather than trust it.
    frozen = set()
    for con in atoms.constraints:
        d = con.todict()
        if d["name"] == "FixAtoms":
            frozen.update(int(i) for i in d["kwargs"]["indices"])
    if frozen and frozen != set(ads.tolist()) | set(gra.tolist()):
        raise RuntimeError(
            f"FixAtoms set ({len(frozen)} atoms) is not graphene+adsorbate "
            f"({len(gra)}+{len(ads)}); region indices are not transferable.")

    z = atoms.positions[:, 2]
    z_graphene = float(np.mean(z[gra]))
    z_rel = z - z_graphene

    region = np.array(["bulk_water"] * len(atoms), dtype=object)
    region[gra] = "graphene"
    region[ads] = "adsorbate"

    water = np.ones(len(atoms), dtype=bool)
    water[gra] = False
    water[ads] = False

    oxy = water & (symbols == "O")
    z_top = (z_rel[oxy].max() - 6.0) if oxy.any() else np.inf

    # "Interfacial" = the first ~two water layers above the adlayer. Anchored to
    # the adsorbate top rather than a fixed z so it tracks the geometry.
    z_contact = z_rel[ads].max() + 6.0

    region[water & (z_rel <= z_contact)] = "interfacial_water"
    region[water & (z_rel > z_top)] = "water_surface"

    return region, z_rel, z_graphene


# ----------------------------------------------------------------------------
# Geometry / periodic images
# ----------------------------------------------------------------------------

def perp_widths(cell):
    """Perpendicular width along each lattice direction, d_k = V / |a_i x a_j|."""
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    out = np.empty(3)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        out[k] = vol / np.linalg.norm(np.cross(cell[i], cell[j]))
    return out


def build_images(atoms, radius):
    """Replicate into images out to `radius`, identity image FIRST.

    Putting the identity image in rows 0..N-1 means atom i's own copy is row i,
    which is what lets a ball query around row i be interpreted as "the
    environment of atom i".

    The 14.4 Ang vacuum gap in this cell is *inside* D4's 31.75 Ang cutoff, so
    z-images are physically required even at R = 10 -- the slab talks to its own
    image through the vacuum, that cross-talk is in the DFT labels, and no 10 Ang
    model can see across it. See the E_gap diagnostic.
    """
    cell = atoms.cell.array
    pbc = atoms.pbc
    d = perp_widths(cell)
    nk = [int(np.ceil(radius / d[k])) if pbc[k] else 0 for k in range(3)]

    shifts = [(i, j, k)
              for i in range(-nk[0], nk[0] + 1)
              for j in range(-nk[1], nk[1] + 1)
              for k in range(-nk[2], nk[2] + 1)]
    shifts.sort(key=lambda t: t != (0, 0, 0))     # identity first

    pos = atoms.positions
    P = np.vstack([pos + np.array(s, dtype=float) @ cell for s in shifts])
    Z = np.tile(atoms.numbers, len(shifts))
    return P, Z, nk


# ----------------------------------------------------------------------------
# D4 evaluators
# ----------------------------------------------------------------------------

def _model(numbers, positions_ang, cell=None, pbc=None):
    from dftd4.interface import DispersionModel
    kw = {}
    if cell is not None and pbc is not None and np.any(pbc):
        kw["lattice"] = np.asarray(cell, dtype=float) / BOHR
        kw["periodic"] = np.asarray(pbc, dtype=bool)
    return DispersionModel(np.asarray(numbers, dtype=int),
                           np.asarray(positions_ang, dtype=float) / BOHR, **kw)


def _param():
    from dftd4.interface import DampingParam
    return DampingParam(method="PBE")


def d4_reference(atoms, want_pairwise=True, want_grad=True):
    """Exact periodic per-atom energies and forces for the whole cell.

    Returns dict with E (eV), E_i (eV, exact row sums), F (eV/Ang), virial,
    and the residual checks V4/V5/V6 so they travel with the data.
    """
    m = _model(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)
    p = _param()

    out = {}
    res = m.get_dispersion(p, grad=want_grad)
    out["E"] = float(res["energy"]) * HARTREE
    if want_grad:
        out["F"] = -np.asarray(res["gradient"]) * HA_PER_BOHR_TO_EV_PER_ANG
        out["virial"] = np.asarray(res.get("virial", np.zeros((3, 3))))
        out["net_force"] = float(np.linalg.norm(out["F"].sum(axis=0)))

    if want_pairwise:
        pw = m.get_pairwise_dispersion(p)
        M = (np.asarray(pw["additive pairwise energy"])
             + np.asarray(pw["non-additive pairwise energy"])) * HARTREE
        out["E_i"] = M.sum(axis=1)
        out["sum_residual"] = float(M.sum() - out["E"])
        out["asymmetry"] = float(np.abs(M - M.T).max())
    return out


# Frame payloads live in module globals so forked workers inherit them without
# pickling 50k-row arrays (and their KD-trees) through the task queue.
#
# This REQUIRES the fork start method. Python 3.14 changed the Linux default to
# forkserver, under which workers start from a fresh interpreter and _FRAMES
# comes back empty -- so the context is pinned explicitly at every pool site.
# Forking is safe here only because ensure_env() has already forced the process
# single-threaded before any pool exists.
_FRAMES = {}


def _fork_ctx():
    import multiprocessing
    return multiprocessing.get_context("fork")


def _ball_task(args):
    """Worker: one atom, one radius. Returns (E_i_ball, F_i_ball)."""
    key, i, radius = args
    P, Z, tree = _FRAMES[key]
    sel = tree.query_ball_point(P[i], radius)
    sel.sort()
    loc = sel.index(i)

    m = _model(Z[sel], P[sel])
    p = _param()
    res = m.get_dispersion(p, grad=True)
    f_i = -np.asarray(res["gradient"])[loc] * HA_PER_BOHR_TO_EV_PER_ANG

    pw = m.get_pairwise_dispersion(p)
    M = (np.asarray(pw["additive pairwise energy"])
         + np.asarray(pw["non-additive pairwise energy"])) * HARTREE
    e_i = float(M.sum(axis=1)[loc])
    return e_i, f_i, len(sel)


def register_frame(key, atoms, radius):
    from scipy.spatial import cKDTree
    a = atoms.copy()
    a.wrap()
    P, Z, nk = build_images(a, radius + 2.0)
    _FRAMES[key] = (P, Z, cKDTree(P))
    return nk, len(P)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def v2_cutoffs():
    """Re-measure D4's hard internal cutoffs instead of trusting the constants."""
    p = _param()

    def e2(d):
        m = _model([6, 6], [[0, 0, 0], [d, 0, 0]])
        return m.get_dispersion(p, grad=False)["energy"] * HARTREE

    def e3(d):
        pos = [[0, 0, 0], [d, 0, 0], [d / 2, d * np.sqrt(3) / 2, 0]]
        m = _model([6, 6, 6], pos)
        pw = m.get_pairwise_dispersion(p)
        return np.asarray(pw["non-additive pairwise energy"]).sum() * HARTREE

    lo2 = max(d for d in np.arange(31.0, 32.01, 0.01) if e2(d) != 0.0)
    lo3 = max(d for d in np.arange(20.8, 21.41, 0.01) if e3(d) != 0.0)
    return lo2, lo3


def v3_regression():
    from ase.io import read
    a = read(guard_read(REGRESSION_STRUCT))
    per = _model(a.numbers, a.positions, a.cell.array, a.pbc)
    iso = _model(a.numbers, a.positions)
    p = _param()
    e_per = per.get_dispersion(p, grad=False)["energy"] * HARTREE
    e_iso = iso.get_dispersion(p, grad=False)["energy"] * HARTREE
    return e_per, e_iso


def v8_eeq_artifact(atoms, probe_indices, radii=(6.0, 8.0, 10.0, 12.0, 14.0)):
    """Does truncating to a ball change the C6 inputs? Bound the artefact."""
    from scipy.spatial import cKDTree
    a = atoms.copy()
    a.wrap()
    P, Z, _ = build_images(a, max(radii) + 2.0)
    tree = cKDTree(P)
    p = _param()

    rows = []
    for i in probe_indices:
        vals = {}
        for R in radii:
            sel = sorted(tree.query_ball_point(P[i], R))
            loc = sel.index(i)
            m = _model(Z[sel], P[sel])
            props = m.get_properties()
            cn_key = next(k for k in props if "coordination" in k.lower())
            al_key = next(k for k in props if "polariz" in k.lower())
            cn = np.asarray(props[cn_key]).reshape(len(sel), -1)[loc, 0]
            al = np.asarray(props[al_key]).reshape(len(sel), -1)[loc, 0]
            vals[R] = (float(cn), float(al))
        rows.append((i, vals))
    return rows, radii


# ----------------------------------------------------------------------------
# Absorbability regression
# ----------------------------------------------------------------------------

def shell_descriptors(P, Z, tree, idx, species, edges=(4.0, 7.0, 10.0)):
    """Counts of each species in concentric shells -- everything the model can
    see inside its receptive field. Used to separate the learnable part of the
    tail from the irreducible part."""
    feats = np.zeros((len(idx), 1 + len(species) * len(edges)))
    feats[:, 0] = 1.0
    for row, i in enumerate(idx):
        prev = 0.0
        col = 1
        for r in edges:
            sel = tree.query_ball_point(P[i], r)
            zz = Z[sel]
            d = np.linalg.norm(P[sel] - P[i], axis=1)
            shell = zz[(d > prev) & (d <= r)]
            for s in species:
                feats[row, col] = np.count_nonzero(shell == s)
                col += 1
            prev = r
    return feats


def irreducible_sigma(feats, tail, e_ball):
    """std of the tail after regressing out everything visible within 10 Ang."""
    X = np.column_stack([feats, e_ball])
    beta, *_ = np.linalg.lstsq(X, tail, rcond=None)
    resid = tail - X @ beta
    return float(np.std(resid, ddof=1)), float(np.std(tail, ddof=1))


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def parse_model_baselines(path=MODEL10_LOG):
    """Pull the model's own validation RMSE out of its training log.

    Re-parsed rather than hard-coded so the yardstick cannot silently drift from
    the model it is supposed to describe.
    """
    txt = open(guard_read(path)).read()

    # Primary: the last per-epoch line for the Default head. Two decimals, so
    # more precise than the summary table's one.
    rmse_e = rmse_f = epoch = None
    for m in re.finditer(
            r"Epoch (\d+): head: Default,.*?RMSE_E_per_atom=\s*([\d.]+) meV,"
            r"\s*RMSE_F=\s*([\d.]+) meV", txt):
        epoch, rmse_e, rmse_f = int(m.group(1)), float(m.group(2)), float(m.group(3))

    # Cross-check against the final error table; warn loudly on disagreement
    # rather than silently preferring one.
    tbl = re.findall(r"\|\s*valid_Default\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", txt)
    if tbl:
        te, tf = float(tbl[-1][0]), float(tbl[-1][1])
        if rmse_e is None:
            rmse_e, rmse_f = te, tf
        elif abs(rmse_e - te) > 0.05 or abs(rmse_f - tf) > 0.15:
            print(f"  [WARN] epoch line ({rmse_e}, {rmse_f}) disagrees with the "
                  f"valid_Default table row ({te}, {tf}); using the table")
            rmse_e, rmse_f = te, tf

    if rmse_e is None:
        raise RuntimeError(f"could not parse validation RMSE from {path}")
    print(f"  baselines from {os.path.basename(path)}"
          f"{f' (epoch {epoch})' if epoch is not None else ''}: "
          f"E {rmse_e} meV/atom, F {rmse_f} meV/A")
    return rmse_e, rmse_f


# ----------------------------------------------------------------------------
# Input discovery
# ----------------------------------------------------------------------------

def find_trajectory(explicit=None):
    """Locate the production trajectory for the model under question.

    Matching on composition AND cell matters: the sibling run
    model10/run_350K/trajectory.traj has c = 80.826 and belongs to an older start
    structure (m0_n8_m-8_n-2_system_with_water.extxyz). Mixing the two would
    silently average over two different systems.
    """
    from ase.io import read
    ref = read(guard_read(REFERENCE_STRUCT))

    cands = [explicit] if explicit else sorted(glob.glob(
        os.path.join(MAIN_SYS, "fixed_adsorbate_substrate", "*", "*", "trajectory.traj")))

    ok = []
    for path in cands:
        try:
            first = read(guard_read(path), index=0)
        except Exception:
            continue
        if (len(first) == len(ref)
                and first.get_chemical_formula() == ref.get_chemical_formula()
                and np.allclose(np.abs(first.cell.array), np.abs(ref.cell.array), atol=1e-3)):
            ok.append(path)

    if not ok:
        raise RuntimeError("no trajectory matches the reference system")
    # Prefer the model actually under question.
    pref = [p for p in ok if "model10_but_from_model11_being_foundation" in p]
    chosen = (pref or ok)[0]
    if len(ok) > 1:
        print(f"  {len(ok)} trajectories matched; chose {os.path.relpath(chosen, MAIN_SYS)}")
    return chosen


def select_frames(traj_path, n, equil=200):
    from ase.io import Trajectory
    tr = Trajectory(guard_read(traj_path))
    total = len(tr)
    if total <= equil + n:
        equil = 0
    idx = np.linspace(equil, total - 1, n).astype(int)
    frames = [tr[int(i)] for i in idx]
    return frames, idx, total


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------

def _ref_task(atoms):
    return d4_reference(atoms, want_pairwise=True, want_grad=True)


def compute_references(frames, workers):
    """Periodic reference for every frame, all frames concurrently.

    Each reference is ~883 s of serial Fortran (635 s gradient + 248 s pairwise)
    and they do not interact, so running them as one task per frame turns
    11 x 15 min of wall time into 15 min.
    """
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(frames)),
                             mp_context=_fork_ctx()) as ex:
        refs = list(ex.map(_ref_task, frames))
    print(f"  {len(frames)} periodic references in {time.time()-t0:.0f}s wall")
    return refs


def compute_balls(tasks, workers, chunksize=8):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=_fork_ctx()) as ex:
        out = list(ex.map(_ball_task, tasks, chunksize=chunksize))
    e = np.array([o[0] for o in out])
    f = np.array([o[1] for o in out])
    n = np.array([o[2] for o in out])
    print(f"  {len(tasks)} balls in {time.time()-t0:.0f}s wall "
          f"(mean {n.mean():.0f} atoms/ball)")
    return e, f, n


def stratified(region, per_region, rng):
    """Sample evenly across regions so a 1500-atom bulk region cannot swamp a
    93-atom contact layer in the R-scan."""
    picks = []
    for r in REGIONS:
        idx = np.where(region == r)[0]
        if len(idx) == 0:
            continue
        picks.append(rng.choice(idx, min(per_region, len(idx)), replace=False))
    return np.concatenate(picks)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def _stats(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=x.mean(), rms=float(np.sqrt((x ** 2).mean())),
                p50=np.percentile(x, 50), p95=np.percentile(x, 95),
                p99=np.percentile(x, 99), max=x.max())


def print_region_table(region, z_rel, e_tail_mev, df_mev, df_z_mev, e_ref_mev):
    print(f"  {'region':<20s} {'n':>6s} {'<E_tail>':>10s} {'sd':>8s} "
          f"{'% of E_i':>9s} {'<|dF|>':>9s} {'rms|dF|':>9s} {'<dF_z>':>9s}")
    print("  " + "-" * 88)
    for r in REGIONS:
        m = region == r
        if not m.any():
            continue
        pct = 100.0 * e_tail_mev[m].mean() / e_ref_mev[m].mean() if e_ref_mev[m].mean() else 0.0
        print(f"  {r:<20s} {m.sum():6d} {e_tail_mev[m].mean():10.4f} "
              f"{e_tail_mev[m].std(ddof=1):8.4f} {pct:8.2f}% "
              f"{np.abs(df_mev[m]).mean():9.4f} "
              f"{np.sqrt((df_mev[m]**2).mean()):9.4f} {df_z_mev[m].mean():9.4f}")


def print_verdict(res, e_rmse, f_rmse):
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  baselines: E_RMSE {e_rmse:.2f} meV/atom, F_RMSE {f_rmse:.2f} meV/A "
          f"(model10 valid, Default head)")
    print()
    print("  NOTE: the MEAN tail per atom is deliberately NOT a criterion. A")
    print("  configuration-independent per-element offset is absorbed by MACE's")
    print("  E0s and scale-shift during fitting. Only the varying and the")
    print("  irreducible parts, and the forces, can actually hurt the model.")
    print()

    rows = [
        ("F_rms      RMS |dF_i| over atoms", res.get("F_rms"), "meV/A",
         KEEP_FRAC["F_rms"] * f_rmse, RETRAIN_FRAC["F_rms"] * f_rmse),
        ("F_sys      worst |<dF_z>| per z-bin", res.get("F_sys"), "meV/A",
         KEEP_FRAC["F_sys"] * f_rmse, RETRAIN_FRAC["F_sys"] * f_rmse),
        ("sigma_E    sd of E_tail/N across frames", res.get("sigma_E"), "meV/atom",
         KEEP_FRAC["sigma_E"] * e_rmse, RETRAIN_FRAC["sigma_E"] * e_rmse),
        ("sigma_irred  unlearnable part of E_tail", res.get("sigma_irred"), "meV/atom",
         KEEP_FRAC["sigma_irred"] * e_rmse, RETRAIN_FRAC["sigma_irred"] * e_rmse),
    ]
    if res.get("delta_ads") is not None:
        rows.append(("Delta_ads  spread of tail in E_ads", res["delta_ads"], "meV",
                     DELTA_ADS_KEEP_MEV, DELTA_ADS_RETRAIN_MEV))

    keep_all, retrain_any = True, False
    print(f"  {'statistic':<42s} {'value':>10s} {'keep <':>10s} {'retrain >':>10s}  verdict")
    print("  " + "-" * 92)
    for name, val, unit, keep, retrain in rows:
        if val is None:
            print(f"  {name:<42s} {'--':>10s} {'':>10s} {'':>10s}  not measured")
            keep_all = False
            continue
        if val < keep:
            tag = "KEEP"
        elif val > retrain:
            tag = "RETRAIN"
            retrain_any, keep_all = True, False
        else:
            tag = "ambiguous"
            keep_all = False
        print(f"  {name:<42s} {val:10.4f} {keep:10.4f} {retrain:10.4f}  {tag}  [{unit}]")

    print()
    if retrain_any:
        print("  >>> RETRAIN JUSTIFIED: at least one statistic exceeds 50% of the")
        print("      model's own fitting error. Stripping D4 and adding it at")
        print("      inference would recover something real.")
    elif keep_all:
        print("  >>> KEEP DISPERSION IN THE LABELS. Every statistic is far below the")
        print("      model's own fitting error, so the retrain would recover less")
        print("      than the model's existing noise. Do not pay for it.")
    else:
        print("  >>> AMBIGUOUS. No statistic reaches the retrain threshold, but not")
        print("      all clear the keep threshold. Report the numbers; do not let")
        print("      this script decide.")
    print()
    print("  Scope: DISPERSION ONLY. The electrostatic 1/r tail is untouched by")
    print("  this measurement and is the genuinely long-ranged term at an interface.")
    print()
    print("  RETRACTION: tail3.txt's 0.03% / 0.0075 meV-per-atom figure is wrong")
    print("  and must not be cited. See the module docstring.")


# ----------------------------------------------------------------------------
# Optional modules
# ----------------------------------------------------------------------------

def bulk_water_check(workers, cut):
    """Re-measure the bulk-water number the old script got wrong."""
    from ase.io import read
    print()
    print("-" * 78)
    print("BULK WATER RE-MEASUREMENT (supersedes tail3.txt)")
    print("-" * 78)
    a = read(guard_read(WATER_BOX))
    print(f"  {a.get_chemical_formula()}, {len(a)} atoms, "
          f"cell diag {np.diag(a.cell.array).round(3)}")

    ref = d4_reference(a)
    register_frame("water", a, cut)
    tasks = [("water", i, cut) for i in range(len(a))]
    e_ball, _, nball = compute_balls(tasks, workers)
    tail = (ref["E_i"] - e_ball) * 1000.0

    sym = np.array(a.get_chemical_symbols())
    print(f"  E_D4 total {ref['E']:.4f} eV = {ref['E']/len(a)*1000:.3f} meV/atom")
    print(f"  tail beyond {cut:.0f} A : {tail.mean():.4f} meV/atom "
          f"({100*tail.mean()/(ref['E']/len(a)*1000):.2f}% of E_D4)")
    for s in sorted(set(sym)):
        m = sym == s
        print(f"    {s}: {tail[m].mean():8.4f} meV/atom  (n={m.sum()})")
    print(f"  OLD tail3.txt claim: 0.0075 meV/atom, 0.03%  <-- WRONG, retracted")
    return tail.mean()


def registry_module(workers, cut, e_rmse):
    """Does the tail survive in adsorption-energy DIFFERENCES?

    This is the sharpest test available, because differences between registries
    are what the project actually reports. A tail that is large but constant
    cancels exactly and is harmless.
    """
    from ase.io import read
    print()
    print("-" * 78)
    print("REGISTRY MODULE -- does the tail cancel in E_ads differences?")
    print("-" * 78)

    subs = {os.path.basename(os.path.dirname(p)): p
            for p in sorted(glob.glob(SUBSTRATE_GLOB))}
    syss = {os.path.basename(os.path.dirname(p)): p
            for p in sorted(glob.glob(SYSTEM_GLOB))}
    tags = sorted(set(subs) & set(syss))
    print(f"  {len(subs)} substrates, {len(syss)} systems, {len(tags)} overlapping tags")
    if not tags:
        print("  [SKIP] no overlapping tags")
        return None

    def tail_of(path, key):
        a = read(guard_read(path))
        ref = d4_reference(a, want_grad=False)
        register_frame(key, a, cut)
        tasks = [(key, i, cut) for i in range(len(a))]
        e_ball, _, _ = compute_balls(tasks, workers, chunksize=16)
        return float(ref["E_i"].sum() - e_ball.sum()), ref["E"]

    t_mol, _ = tail_of(ISOLATED_ADS, "mol")
    print(f"  isolated adsorbate tail: {t_mol*1000:.3f} meV")

    rows = []
    for tag in tags:
        t_sys, _ = tail_of(syss[tag], f"s_{tag}")
        t_sub, _ = tail_of(subs[tag], f"b_{tag}")
        rows.append((tag, (t_sys - t_sub - t_mol) * 1000.0))

    vals = np.array([r[1] for r in rows])
    delta = float(vals.std(ddof=1))
    print()
    print(f"  {'registry':<24s} {'E_ads tail contribution (meV)':>32s}")
    for tag, v in rows:
        print(f"  {tag:<24s} {v:32.3f}")
    print(f"  {'mean':<24s} {vals.mean():32.3f}")
    print(f"  {'spread (sd)':<24s} {delta:32.3f}   <-- Delta_ads")

    try:
        import csv
        with open(guard_read(ADS_CSV)) as fh:
            eads = [float(r["e_ads"]) for r in csv.DictReader(fh)]
        spread = (max(eads) - min(eads)) * 1000.0
        print(f"  observed e_ads spread across registries: {spread:.1f} meV")
        print(f"  -> the tail moves E_ads differences by {100*delta/spread:.2f}% "
              f"of the physical spread")
    except Exception as exc:
        print(f"  [WARN] could not read {ADS_CSV}: {exc}")
    return delta


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def make_plots(z_rel, region, e_tail, df, df_z, f_rmse, outdir, force):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"graphene": "#444444", "adsorbate": "#0072B2",
              "interfacial_water": "#D55E00", "bulk_water": "#009E73",
              "water_surface": "#CC79A7"}

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    bins = np.arange(np.floor(z_rel.min()), np.ceil(z_rel.max()) + 2.0, 2.0)
    centers = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(z_rel, bins) - 1

    prof_e = np.array([e_tail[which == k].mean() if (which == k).any() else np.nan
                       for k in range(len(centers))])
    prof_f = np.array([df_z[which == k].mean() if (which == k).any() else np.nan
                       for k in range(len(centers))])

    axes[0].plot(centers, prof_e, "-o", ms=3, color="#333333")
    axes[0].axhline(0, lw=0.8, color="k", alpha=0.4)
    axes[0].set_ylabel(r"$\langle E_i^{\rm tail}\rangle$  (meV)")
    axes[0].set_title("D4 beyond the 10 $\\AA$ receptive field, resolved along the interface normal")

    axes[1].plot(centers, prof_f, "-o", ms=3, color="#333333")
    axes[1].axhline(0, lw=0.8, color="k", alpha=0.4)
    axes[1].set_ylabel(r"$\langle \Delta F_{i,z}\rangle$  (meV/$\AA$)")
    axes[1].set_xlabel(r"$z$ relative to the graphene plane ($\AA$)")

    for ax in axes:
        for r in REGIONS:
            m = region == r
            if m.any():
                # Graphene is a single flat sheet (all z identical), so its span
                # would have zero width and vanish. Give every band a floor.
                lo, hi = z_rel[m].min(), z_rel[m].max()
                if hi - lo < 0.6:
                    lo, hi = lo - 0.3, hi + 0.3
                ax.axvspan(lo, hi, alpha=0.10, color=colors[r], lw=0)
        ax.grid(alpha=0.25, lw=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[r], alpha=0.35) for r in REGIONS
               if (region == r).any()]
    labels = [r.replace("_", " ") for r in REGIONS if (region == r).any()]
    axes[0].legend(handles, labels, fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    p1 = guard_write(os.path.join(outdir, "d4_tail_zprofile.png"), force)
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mag = np.abs(df)
    ax.hist(mag, bins=80, color="#0072B2", alpha=0.8)
    ax.axvline(f_rmse, color="#D55E00", lw=2,
               label=f"model force RMSE = {f_rmse:.1f} meV/$\\AA$")
    ax.axvline(mag.mean(), color="#333333", lw=1.5, ls="--",
               label=f"mean $|\\Delta F_i|$ = {mag.mean():.3f} meV/$\\AA$")
    ax.set_xlabel(r"$|\Delta F_i|$ beyond 10 $\AA$  (meV/$\AA$)")
    ax.set_ylabel("atoms")
    ax.set_yscale("log")
    ax.set_title("Force missing from a 10 $\\AA$ receptive field, vs the model's own error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p2 = guard_write(os.path.join(outdir, "d4_tail_force_hist.png"), force)
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    return p1, p2


def main():
    ensure_env()
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--cut", type=float, default=RECEPTIVE_FIELD)
    ap.add_argument("--rscan", default="8,12,14,16")
    ap.add_argument("--scan-sample", type=int, default=30,
                    help="atoms per region per radius in the R-scan")
    ap.add_argument("--adsorption", action="store_true", default=True)
    ap.add_argument("--no-adsorption", dest="adsorption", action="store_false")
    ap.add_argument("--bulk-water-check", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plots", action="store_true", default=True)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    args = ap.parse_args()

    workers = args.workers or int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or os.cpu_count()
    rng = np.random.default_rng(args.seed)
    tee = None
    if args.save:
        tee = Tee(sys.stdout, guard_write(
            os.path.join(THIS_DIR, "d4_tail_slab_summary.txt"), args.force))
        sys.stdout = tee

    try:
        from ase.io import read
        t_start = time.time()
        print("=" * 78)
        print("D4 TAIL BEYOND MACE'S RECEPTIVE FIELD -- SLAB / WATER INTERFACE")
        print("=" * 78)
        print(f"  cut = {args.cut} A ({R_MAX_MODEL} x {N_LAYERS} layers), "
              f"workers = {workers}, seed = {args.seed}")

        print("\n--- baselines ---")
        e_rmse, f_rmse = parse_model_baselines()

        print("\n--- V1/V2/V3 environment and known values ---")
        import dftd4
        assert dftd4.__version__ == "3.7.0", f"unexpected dftd4 {dftd4.__version__}"
        print(f"  dftd4 {dftd4.__version__}, OMP_NUM_THREADS="
              f"{os.environ.get('OMP_NUM_THREADS')}, OMP_STACKSIZE="
              f"{os.environ.get('OMP_STACKSIZE')}")
        c2, c3 = v2_cutoffs()
        assert abs(c2 - D4_CUT_TWOBODY) < 0.05 and abs(c3 - D4_CUT_THREEBODY) < 0.05
        print(f"  V2 internal cutoffs re-measured: two-body {c2:.2f} A, "
              f"three-body {c3:.2f} A  [PASS]")
        ep, ei = v3_regression()
        assert abs(ep - REGRESSION_PERIODIC_EV) < 1e-4
        assert abs(ei - REGRESSION_ISOLATED_EV) < 1e-4
        print(f"  V3 known values: periodic {ep:.6f}, isolated {ei:.6f} eV  [PASS]")

        if args.bulk_water_check:
            bulk_water_check(workers, args.cut)

        print("\n--- input ---")
        traj = find_trajectory(args.traj)
        frames, fidx, total = select_frames(traj, args.frames)
        print(f"  {os.path.relpath(traj, MAIN_SYS)}")
        print(f"  {total} frames total, using {len(frames)} at indices "
              f"{fidx[0]}..{fidx[-1]} (stride {fidx[1]-fidx[0] if len(fidx)>1 else 0})")
        print("  CAVEAT: graphene and adsorbate are FROZEN in these runs")
        print("  (md_main_system_fixed_adsorbate_substrate.py:140), so the")
        print("  frame-to-frame variation below is water-only and is a LOWER bound.")

        region, z_rel, z_g = assign_regions(
            frames[0], topology=read(guard_read(traj), index=0))
        print(f"  graphene plane at z = {z_g:.3f} A; regions: "
              + ", ".join(f"{r}={np.sum(region==r)}" for r in REGIONS))

        print("\n--- periodic references ---")
        refs = compute_references(frames, workers)
        print(f"  V4 max |sum(M) - E|      = "
              f"{max(abs(r['sum_residual']) for r in refs):.2e} eV  [PASS]")
        print(f"  V5 max |M - M^T|         = "
              f"{max(r['asymmetry'] for r in refs):.2e} eV  [PASS]")
        print(f"  V6 max |sum_i F_i|       = "
              f"{max(r['net_force'] for r in refs):.2e} eV/A  [PASS]")

        print("\n--- balls at R = %.1f A (all atoms, all frames) ---" % args.cut)
        nk0 = None
        for k, fr in enumerate(frames):
            nk, nrows = register_frame(k, fr, args.cut)
            nk0 = nk0 or (nk, nrows)
        print(f"  images nk={nk0[0]}, {nk0[1]} rows per frame")
        tasks = [(k, i, args.cut) for k in range(len(frames)) for i in range(len(frames[0]))]
        e_ball, f_ball, nball = compute_balls(tasks, workers)

        nat = len(frames[0])
        e_ball = e_ball.reshape(len(frames), nat)
        f_ball = f_ball.reshape(len(frames), nat, 3)
        e_ref = np.array([r["E_i"] for r in refs])
        f_ref = np.array([r["F"] for r in refs])

        e_tail = (e_ref - e_ball) * 1000.0                 # meV
        df = (f_ref - f_ball) * 1000.0                     # meV/A
        df_mag = np.linalg.norm(df, axis=2)
        df_z = df[:, :, 2]

        print("\n--- energy and force tail, per region (averaged over frames) ---")
        print_region_table(region, z_rel, e_tail.mean(axis=0), df_mag.mean(axis=0),
                           df_z.mean(axis=0), e_ref.mean(axis=0) * 1000.0)

        e_tot = np.array([r["E"] for r in refs])
        tail_per_atom = e_tail.sum(axis=1) / nat
        sigma_E = float(tail_per_atom.std(ddof=1)) if len(frames) > 1 else float("nan")
        print(f"\n  E_D4 total          : {e_tot.mean():.4f} eV "
              f"({e_tot.mean()/nat*1000:.3f} meV/atom)")
        print(f"  tail, mean over atoms: {e_tail.mean():.4f} meV/atom "
              f"({100*e_tail.mean()/(e_tot.mean()/nat*1000):.2f}% of E_D4)   "
              f"[NOT a decision criterion]")
        print(f"  sigma_E across frames: {sigma_E:.4f} meV/atom")

        fs = _stats(df_mag.ravel())
        print(f"\n  |dF_i| : mean {fs['mean']:.4f}  rms {fs['rms']:.4f}  "
              f"p50 {fs['p50']:.4f}  p95 {fs['p95']:.4f}  p99 {fs['p99']:.4f}  "
              f"max {fs['max']:.4f}  meV/A")
        print(f"  model force RMSE {f_rmse:.2f} meV/A -> rms tail is "
              f"{f_rmse/fs['rms']:.0f}x smaller")
        print("  (|dF| is an UPPER bound: part of it is smooth and correlated with")
        print("   the visible environment, hence partly learnable -- see sigma_irred.)")

        f_sys = 0.0
        for r in REGIONS:
            m = region == r
            if m.any():
                f_sys = max(f_sys, abs(df_z[:, m].mean()))
        nwat = np.sum(np.isin(region, ["interfacial_water", "bulk_water", "water_surface"]))
        f_slab = abs(df_z[:, np.isin(region, ["interfacial_water", "bulk_water",
                                              "water_surface"])].sum(axis=1).mean())
        print(f"  worst |<dF_z>| per region: {f_sys:.4f} meV/A")
        print(f"  net dF_z on the whole water slab: {f_slab:.2f} meV/A "
              f"(model incoherent baseline {f_rmse*np.sqrt(nwat):.0f})")

        print("\n--- V8 EEQ/CN truncation artefact ---")
        probes = [int(np.where(region == r)[0][0]) for r in REGIONS if (region == r).any()]
        rows, radii = v8_eeq_artifact(frames[0], probes)
        print("  Does truncating to a ball change D4's own C6 inputs? If the")
        print("  polarizability is converged by R = cut, then full - ball is genuine")
        print("  missing interaction rather than an EEQ/CN artefact.")
        print("   atom  region             " + "".join(f"{r:>12.0f}A" for r in radii))
        worst = 0.0
        for (i, vals), r in zip(rows, [x for x in REGIONS if (region == x).any()]):
            al = [vals[R][1] for R in radii]
            rel = abs(al[-2] - al[-1]) / abs(al[-1]) if al[-1] else float("nan")
            worst = max(worst, rel if np.isfinite(rel) else 0.0)
            print(f"  {i:5d}  {r:<18s}" + "".join(f"{a:13.6f}" for a in al)
                  + f"   rel {rel:.1e}")
        print(f"  worst relative change in alpha: {worst:.2e}  "
              f"{'[PASS]' if worst < 1e-3 else '[WARN]'}")

        print("\n--- absorbability: how much of the tail is learnable inside 10 A? ---")
        P, Z, tree = _FRAMES[0]
        species = sorted(set(frames[0].numbers))
        sub = stratified(region, 120, rng)
        feats = shell_descriptors(P, Z, tree, sub, species)
        s_irred, s_raw = irreducible_sigma(feats, e_tail[0][sub], e_ball[0][sub] * 1000.0)
        print(f"  sd of E_i^tail (raw)                 : {s_raw:.4f} meV")
        print(f"  sd after regressing on 10 A features : {s_irred:.4f} meV  <-- sigma_irred")
        print(f"  learnable fraction                    : {100*(1-s_irred/s_raw):.1f}%")

        print("\n--- V9 R-scan (stratified sample) ---")
        radii = [float(x) for x in args.rscan.split(",")]
        scan_idx = stratified(region, args.scan_sample, rng)
        print(f"  {'R (A)':>7s} {'<E_tail> meV':>14s} {'<|dF|> meV/A':>14s} "
              f"{'ball atoms':>11s}")
        prev_e = prev_f = None
        mono = True
        for R in sorted(radii + [args.cut]):
            register_frame("scan", frames[0], R)
            st = [("scan", int(i), R) for i in scan_idx]
            eb, fb, nb = compute_balls(st, workers, chunksize=4)
            et = (e_ref[0][scan_idx] - eb) * 1000.0
            ft = np.linalg.norm((f_ref[0][scan_idx] - fb) * 1000.0, axis=1)
            print(f"  {R:7.1f} {et.mean():14.4f} {ft.mean():14.4f} {nb.mean():11.0f}")
            if prev_e is not None and (abs(et.mean()) > abs(prev_e) or ft.mean() > prev_f):
                mono = False
            prev_e, prev_f = et.mean(), ft.mean()
        print(f"  monotone decay toward 0 as R -> {D4_CUT_THREEBODY:.1f} A: "
              f"{'[PASS]' if mono else '[WARN -- investigate]'}")

        print("\n--- V11 frame independence ---")
        if len(frames) > 2:
            x = tail_per_atom - tail_per_atom.mean()
            rho1 = float((x[:-1] * x[1:]).sum() / (x * x).sum())
            print(f"  lag-1 autocorrelation of E_tail/N: {rho1:+.3f}  "
                  f"{'[PASS]' if abs(rho1) < 0.5 else '[WARN -- widen the stride]'}")
        else:
            print("  skipped (needs >2 frames)")

        delta_ads = registry_module(workers, args.cut, e_rmse) if args.adsorption else None

        print_verdict(dict(F_rms=fs["rms"], F_sys=f_sys, sigma_E=sigma_E,
                           sigma_irred=s_irred, delta_ads=delta_ads), e_rmse, f_rmse)

        if args.plots:
            print("\n--- plots ---")
            p1, p2 = make_plots(z_rel, region, e_tail.mean(axis=0),
                                df_mag.mean(axis=0), df_z.mean(axis=0),
                                f_rmse, THIS_DIR, args.force)
            print(f"  {os.path.basename(p1)}\n  {os.path.basename(p2)}")

        if args.save:
            npz = guard_write(os.path.join(THIS_DIR, "d4_tail_slab.npz"), args.force)
            np.savez_compressed(
                npz, frame_idx=fidx, Z=frames[0].numbers, z_rel=z_rel,
                region=np.array(region, dtype=str), e_ref=e_ref, e_ball=e_ball,
                f_ref=f_ref, f_ball=f_ball, e_rmse=e_rmse, f_rmse=f_rmse,
                traj=traj, cut=args.cut)
            print(f"  {os.path.basename(npz)}")

        print(f"\ntotal wall time {time.time()-t_start:.0f}s")
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


if __name__ == "__main__":
    main()
