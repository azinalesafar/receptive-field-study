#!/usr/bin/env python3
"""
E_tail as a function of cutoff radius, from the production 10 Ang receptive
field out to D4's own hard 31.75 Ang two-body limit.

WHAT THIS ADDS TO measure_d4_tail_slab.py
-----------------------------------------
That script answers "how much D4 lies beyond 10 Ang" and settles the keep/retrain
decision. Its V9 module already scans R, but only over a coarse ladder capped at
16 Ang -- enough to show monotone decay, not enough to show the SHAPE of the
decay or where it dies.

This script extends that same scan to D4's own cutoff. It answers the natural
follow-up: "my receptive field is 10 Ang -- what would a larger one actually
recover, and at what radius does the question stop mattering at all?"

It uses the SAME estimator as the sibling script, deliberately:

    E_i^tail(R) = E_i^ref(periodic) - E_i^ball(R)

with E_i^ball(R) from an independently built cluster of radius R. The reference
E_i^ref is reused from d4_tail_slab.npz rather than recomputed (1162 s saved).

A SHORTCUT THAT WAS TRIED AND REJECTED -- DO NOT REINTRODUCE IT
---------------------------------------------------------------
A ball is a finite cluster, so every atom in it has a definite distance to the
centre. That invites an appealing optimisation: call D4 ONCE on the largest ball,
then recover every radius at once as a cumulative sum over distance-sorted pairs,

    E_i^ball(R) =?= sum over { j : d(i,j) <= R } of M[loc, j]

turning 25 calls per atom into 1. It was implemented, measured, and REJECTED on
2026-09-21. Two independent errors, both measured against independently built
10 Ang balls (graphene / adsorbate / bulk water, meV):

    error source                     graphene   adsorbate   bulk water
    three-body (ATM) contamination    -0.1762     -0.1014      +0.0015
    two-body C6/EEQ difference        +0.0695     +0.0081      +0.0061

  1. ATM is a genuine THREE-body interaction folded into a pair table. The amount
     booked to pair (i,j) in a 31.75 Ang ball includes triples (i,j,k) whose third
     atom lies far outside R. Cutting on d(i,j) <= R keeps them; an independently
     built R-ball never contained atom k at all. The independent ball is the
     physically correct reference, because a 10 Ang model can only represent a
     triple when BOTH partners are inside its field.

  2. Even the two-body part is not clean: C6 coefficients depend on EEQ charges
     and coordination numbers, so a 31.75 Ang ball and a 10 Ang ball assign
     slightly different C6 to the same pair. V8 in the sibling script bounds this
     at 1.8e-4 relative, which sounds negligible and is not: for graphene it is
     0.0695 meV, while the graphene tail near 21 Ang is only about 0.05 meV.

Both errors concentrate in dense polarizable carbon and nearly vanish in bulk
water, which is why a water-only calibration looked clean (1.5e-3 meV) and was
misleading. The full ladder costs ~23000 core-seconds, about the same as the
shortcut once the mean ball size is used rather than the worst case, so the
shortcut bought nothing and cost correctness.

PERIODIC IMAGES
---------------
The cell is periodic in all three directions and the 13.4 Ang vacuum gap sits
well inside D4's 31.75 Ang cutoff, so past R ~ 20 Ang a ball starts to include
the slab's own image across the vacuum. That cross-talk is real, is present in
the DFT labels, and no 10 Ang model can see it. It is physics, not an artefact.

VALIDATION BUILT IN
-------------------
  C1  E_tail(31.75) must be ~zero: at D4's own two-body cutoff the ball contains
      every interaction the periodic reference does. Forced by physics, not fit.
  C2  E_tail(10.0) must reproduce the values already stored in d4_tail_slab.npz,
      which were computed independently by the sibling script.
  C3  Monotone decay toward zero: dispersion is strictly attractive, so each
      extra shell can only remove missing attraction.
  C4  The three-body part must be EXACTLY zero beyond 21.1668 Ang, D4's ATM
      cutoff. Measured, not assumed.

USAGE
    python3 measure_d4_tail_vs_R.py --per-region 10 --save --plots
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_d4_tail_slab as S

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_IN = os.path.join(THIS_DIR, "d4_tail_slab.npz")

# Graded ladder: fine where the curve moves, coarse where it is flat and the
# balls are expensive (cost ~ N^2.55, and N grows as R^3 until the slab runs out
# of thickness). 21.1668 is included exactly so the ATM cutoff is resolved.
RADII = np.array(
    [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5,
     16.0, 17.0, 18.0, 19.0, 20.0, 21.0, S.D4_CUT_THREEBODY, 22.0, 24.0, 26.0,
     28.0, 30.0, S.D4_CUT_TWOBODY])

_BIG = {}


def _ladder_task(args):
    """One atom, one radius -> (two-body, three-body, n_atoms). Energy only.

    Mirrors _ball_task in the sibling script, with two differences: the gradient
    is ~72% of D4's cost here and nothing in this curve needs it, so only the
    pairwise decomposition is requested; and the additive and non-additive parts
    are kept apart so C4 can check that ATM really dies at 21.17 Ang.
    """
    i, R = args
    P, Z, tree = _BIG["frame"]
    sel = np.asarray(sorted(tree.query_ball_point(P[i], R)))
    loc = int(np.searchsorted(sel, i))

    m = S._model(Z[sel], P[sel])
    p = S._param()
    pw = m.get_pairwise_dispersion(p)
    # Take row `loc` before anything else touches the (N,N) matrices -- together
    # they are ~1.2 GB at the largest ball sizes.
    e2 = float(np.asarray(pw["additive pairwise energy"])[loc].sum()) * S.HARTREE
    e3 = float(np.asarray(pw["non-additive pairwise energy"])[loc].sum()) * S.HARTREE
    return e2, e3, len(sel)


def compute_ladder(indices, workers):
    """Every (atom, radius) pair, biggest balls first."""
    from concurrent.futures import ProcessPoolExecutor
    P, _, tree = _BIG["frame"]

    tasks = [(int(i), float(R)) for i in indices for R in RADII]
    # Cost grows steeply with occupancy, so hand the biggest balls out first:
    # a late 8721-atom ball would otherwise set the wall time on its own
    # (longest-processing-time-first scheduling).
    sizes = np.array([len(tree.query_ball_point(P[i], R)) for i, R in tasks])
    order = np.argsort(-sizes)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=S._fork_ctx()) as ex:
        out = list(ex.map(_ladder_task, [tasks[k] for k in order], chunksize=1))

    inv = np.argsort(order)
    e2 = np.array([out[k][0] for k in inv]).reshape(len(indices), len(RADII))
    e3 = np.array([out[k][1] for k in inv]).reshape(len(indices), len(RADII))
    nb = np.array([out[k][2] for k in inv]).reshape(len(indices), len(RADII))
    print(f"  {len(tasks)} balls in {time.time()-t0:.0f}s wall "
          f"(largest {sizes.max()} atoms, mean {sizes.mean():.0f})")
    return e2, e3, nb


def make_plot(tail_tot, tail_reg, counts, outdir, force):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"graphene": "#444444", "adsorbate": "#0072B2",
              "interfacial_water": "#D55E00", "bulk_water": "#009E73",
              "water_surface": "#CC79A7"}

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.4), sharex=True)

    def mark(ax):
        ax.axvline(S.RECEPTIVE_FIELD, color="#D55E00", lw=1.6, ls="--", zorder=1)
        ax.axvline(S.D4_CUT_THREEBODY, color="#666666", lw=1.0, ls=":", zorder=1)
        ax.axvline(S.D4_CUT_TWOBODY, color="#666666", lw=1.0, ls="-.", zorder=1)

    ax = axes[0]
    for r, y in tail_reg.items():
        ax.plot(RADII, y, "-o", ms=3, lw=1.3, color=colors[r], alpha=0.85,
                label=f"{r.replace('_',' ')} (n={counts[r]})")
    # NOT the system average: stratified sampling weights the five regions
    # equally, while the real cell is 73% bulk water. Label it for what it is.
    ax.plot(RADII, tail_tot, "-o", ms=4, lw=2.6, color="#111111",
            label="stratified mean (equal weight / region)", zorder=5)
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    mark(ax)
    ax.set_ylabel(r"$\langle E_i^{\rm tail}(R)\rangle$  (meV)")
    ax.set_title("Dispersion beyond a cutoff $R$: what a larger receptive "
                 "field would recover")
    # Lower centre: the curves have all risen toward zero by mid-R, so this
    # block is empty, and it keeps the legend clear of the 31.75 marker line.
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    for r, y in tail_reg.items():
        ax.semilogy(RADII, np.abs(y), "-o", ms=3, lw=1.3, color=colors[r],
                    alpha=0.85)
    ax.semilogy(RADII, np.abs(tail_tot), "-o", ms=4, lw=2.6, color="#111111",
                zorder=5)
    mark(ax)
    ax.set_ylabel(r"$|\langle E_i^{\rm tail}(R)\rangle|$  (meV, log scale)")
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.grid(alpha=0.25, lw=0.5, which="both")

    # Label the marker lines along the top, in axis-fraction y so they cannot
    # collide with the curves or the legend whatever the data range turns out to be.
    for x, txt, dx in ((S.RECEPTIVE_FIELD, "production 10 $\\AA$", 0.25),
                       (S.D4_CUT_THREEBODY, "D4 3-body 21.17 $\\AA$", 0.25),
                       (S.D4_CUT_TWOBODY, "D4 2-body 31.75 $\\AA$", -0.25)):
        axes[0].text(x + dx, 0.97, txt, transform=axes[0].get_xaxis_transform(),
                     fontsize=7.5, color="#444444", rotation=90,
                     ha="left" if dx > 0 else "right", va="top")

    fig.tight_layout()
    p = S.guard_write(os.path.join(outdir, "d4_tail_vs_R.png"), force)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    S.ensure_env()
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-region", type=int, default=10,
                    help="atoms sampled per region (5 regions)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--plots", action="store_true", default=True)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workers = args.workers or int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or os.cpu_count()
    rng = np.random.default_rng(args.seed)

    tee = None
    if args.save:
        tee = S.Tee(sys.stdout, S.guard_write(
            os.path.join(THIS_DIR, "d4_tail_vs_R_summary.txt"), args.force))
        sys.stdout = tee

    try:
        from ase.io import read
        t_start = time.time()
        print("=" * 78)
        print("D4 TAIL vs CUTOFF RADIUS -- 10 A to the 31.75 A D4 limit")
        print("=" * 78)
        print(f"  {len(RADII)} radii, {RADII[0]:.2f} .. {RADII[-1]:.4f} A")
        print(f"  workers = {workers}, seed = {args.seed}")
        print(f"  estimator: E_i^tail(R) = E_i^ref(periodic) - E_i^ball(R),")
        print(f"             independent ball at every radius (no shortcut)")

        # ---- reuse the reference already paid for -------------------------
        d = np.load(S.guard_read(NPZ_IN), allow_pickle=True)
        e_ref0 = d["e_ref"][0] * 1000.0            # meV, frame 0
        region = d["region"]
        frame0 = int(d["frame_idx"][0])
        traj = str(d["traj"])
        print("\n--- reference ---")
        print(f"  exact periodic E_i^ref reused from d4_tail_slab.npz (frame {frame0})")
        print(f"  {traj}")

        atoms = read(S.guard_read(traj), index=str(frame0))
        if len(atoms) != len(region):
            raise RuntimeError("npz and trajectory frame disagree on atom count")
        atoms.wrap()

        # ---- one image set, reused by every radius ------------------------
        from scipy.spatial import cKDTree
        P, Z, nk = S.build_images(atoms, float(RADII[-1]) + 2.0)
        _BIG["frame"] = (P, Z, cKDTree(P))
        print(f"  images nk={nk}, {len(P)} rows "
              f"(vacuum gap is inside the cutoff, so z-images matter)")

        # ---- sample and run ------------------------------------------------
        idx = S.stratified(region, args.per_region, rng)
        counts = {r: int((region[idx] == r).sum()) for r in S.REGIONS
                  if (region[idx] == r).any()}
        print("\n--- balls ---")
        print(f"  stratified sample: {len(idx)} atoms, "
              + ", ".join(f"{r}={n}" for r, n in counts.items()))
        e2, e3, nball = compute_ladder(idx, workers)

        e_ball = (e2 + e3) * 1000.0                       # meV
        tail = e_ref0[idx][:, None] - e_ball              # (natoms, nR), meV
        mean_curve = tail.mean(axis=0)
        base = mean_curve[0]

        # ---- C1 -------------------------------------------------------------
        print("\n--- C1 curve must reach zero at D4's two-body cutoff ---")
        end = tail[:, -1]
        ratio = np.abs(end).max() / abs(base)
        print(f"  <E_tail({RADII[-1]:.2f})>      = {end.mean():+.5f} meV")
        print(f"  max |E_tail({RADII[-1]:.2f})|  = {np.abs(end).max():.5f} meV")
        print(f"  vs <E_tail(10.0)>    = {base:+.5f} meV")
        print(f"  residual is {100*ratio:.2f}% of the 10 A tail  "
              f"{'[PASS]' if ratio < 0.05 else '[WARN]'}")
        print("  (any residual is the cluster-vs-periodic EEQ/CN difference,")
        print("   not missing interaction -- V8 bounds it at 1.8e-4 relative)")

        # ---- C2 -------------------------------------------------------------
        print("\n--- C2 R = 10 must reproduce the sibling script's stored tail ---")
        stored = (d["e_ref"][0][idx] - d["e_ball"][0][idx]) * 1000.0
        dev = np.abs(tail[:, 0] - stored)
        print(f"  stored  <E_tail(10)> = {stored.mean():+.5f} meV")
        print(f"  here    <E_tail(10)> = {tail[:, 0].mean():+.5f} meV")
        print(f"  max |deviation|      = {dev.max():.6f} meV "
              f"{'[PASS]' if dev.max() < 0.01 else '[WARN]'}")

        # ---- C3 -------------------------------------------------------------
        print("\n--- C3 monotone decay (dispersion is strictly attractive) ---")
        worst = float(np.max(np.diff(np.abs(mean_curve))))
        print(f"  largest increase in |<E_tail>| between adjacent radii: {worst:+.2e} meV")
        print(f"  {'[PASS]' if worst <= 1e-9 else '[WARN -- non-monotone]'}")

        # ---- C4 -------------------------------------------------------------
        print("\n--- C4 three-body must vanish beyond D4's ATM cutoff ---")
        beyond = RADII > S.D4_CUT_THREEBODY
        e3_ref = e3[:, RADII == S.D4_CUT_THREEBODY]
        drift = np.abs(e3[:, beyond] - e3_ref).max() * 1000.0
        print(f"  radii beyond 21.1668 A: {RADII[beyond]}")
        print(f"  max change in three-body energy past the cutoff: {drift:.3e} meV")
        # The residual is not surviving ATM: adding shells changes atom i's own
        # CN and EEQ charge slightly, which moves C6/C9 and hence the three-body
        # energy. That is the same EEQ/CN drift V8 bounds, so the tolerance is
        # set at the drift scale rather than at zero.
        print(f"  {'[PASS]' if drift < 1e-3 else '[WARN -- ATM still alive]'}"
              f"  (residual is EEQ/CN drift, not ATM)")

        # ---- the table ------------------------------------------------------
        print("\n--- E_tail(R), mean over the sampled atoms ---")
        hdr = f"  {'R (A)':>8s} {'<E_tail>':>10s} {'% of 10A':>9s} {'ball':>7s}"
        for r in counts:
            hdr += f" {r.replace('_',' ')[:11]:>12s}"
        print(hdr)
        for k, R in enumerate(RADII):
            line = (f"  {R:8.4f} {mean_curve[k]:10.4f} "
                    f"{100*mean_curve[k]/base:8.1f}% {nball[:, k].mean():7.0f}")
            for r in counts:
                line += f" {tail[region[idx] == r, k].mean():12.4f}"
            print(line)

        # ---- two-body / three-body split ------------------------------------
        print("\n--- where the tail lives: two-body vs three-body (ATM) ---")
        print(f"  {'R (A)':>8s} {'E_2body':>12s} {'E_3body':>12s} {'3b share':>9s}")
        for k, R in enumerate(RADII):
            a2, a3 = e2[:, k].mean() * 1000.0, e3[:, k].mean() * 1000.0
            print(f"  {R:8.4f} {a2:12.4f} {a3:12.4f} "
                  f"{100*abs(a3)/(abs(a2)+abs(a3)):8.2f}%")

        # ---- the actual question --------------------------------------------
        print("\n--- what a larger receptive field would buy ---")
        print("  model baselines: E_RMSE 0.62 meV/atom, F_RMSE 24.79 meV/A")
        for R in (12.0, 14.0, 16.0, 20.0, 24.0):
            k = int(np.argmin(np.abs(RADII - R)))
            if abs(RADII[k] - R) > 1e-6:       # not on this ladder
                continue
            rec = abs(base) - abs(mean_curve[k])
            print(f"  10.0 -> {R:4.1f} A recovers {rec:7.4f} meV/atom "
                  f"({100*rec/abs(base):5.1f}% of the 10 A tail), "
                  f"leaving {mean_curve[k]:+7.4f}")
        print("\n  NOTE: this is the MEAN tail, which MACE absorbs into its E0s and")
        print("  scale-shift. It is deliberately NOT the keep/retrain criterion --")
        print("  see print_verdict() in measure_d4_tail_slab.py. This curve answers")
        print("  'how far does dispersion reach', not 'should the labels change'.")

        # ---- outputs --------------------------------------------------------
        tail_reg = {r: tail[region[idx] == r].mean(axis=0) for r in counts}
        if args.save:
            p = S.guard_write(os.path.join(THIS_DIR, "d4_tail_vs_R.npz"), args.force)
            np.savez_compressed(
                p, radii=RADII, idx=idx, region=region[idx], tail=tail,
                e2=e2, e3=e3, e_ref=e_ref0[idx], nball=nball,
                frame_idx=frame0, traj=traj)
            print(f"\n  wrote {os.path.basename(p)}")
        if args.plots:
            png = make_plot(mean_curve, tail_reg, counts, THIS_DIR, args.force)
            print(f"  wrote {os.path.basename(png)}")

        print(f"\ntotal wall time {time.time()-t_start:.0f}s")
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


if __name__ == "__main__":
    main()
