#!/usr/bin/env python3
"""
Whole-system E_tail as a function of cutoff radius, using EVERY atom of one frame.

WHAT THIS ADDS TO measure_d4_tail_vs_R.py
-----------------------------------------
That script samples 10 atoms per region and weights the five regions equally.
Two problems followed: the curve is not the system average (the real cell is
~73% bulk water), and the small samples were element-biased (its 10
interfacial-water atoms were all H, so that curve came out spuriously low).

Here every atom of the frame is used, so at each radius there is one number
for the whole system,

    <E_tail(R)> = (1/N) sum_i [ E_i^ref(periodic) - E_i^ball(R) ]

with no sampling at all. Region and region x element curves come for free.

The estimator, the radius ladder and the per-ball task are imported unchanged
from measure_d4_tail_vs_R.py (independent ball at every radius; the one-call
shortcut is WRONG for the three-body term -- see that module's docstring).

CHECKPOINTING
-------------
~52000 D4 calls take hours, so finished balls are written to checkpoint.npz
every --checkpoint-every seconds. Re-running the same command resumes from it.

VALIDATION BUILT IN
-------------------
  C1  <E_tail(31.75)> must be ~zero (D4's own two-body cutoff).
  C2  E_tail(10.0) must reproduce, atom by atom, the values stored in
      d4_tail_slab.npz for the same frame (computed independently there).
  C3  The whole-system curve must decay monotonically toward zero.

USAGE
    python3 measure_d4_tail_vs_R_all_atoms.py --workers 128 --save
    python3 measure_d4_tail_vs_R_all_atoms.py --workers 128 --save --frame-slot 5   # frame 933
    python3 measure_d4_tail_vs_R_all_atoms.py --test      # 3 atoms, 3 radii
"""
import os
import sys
import time
import argparse

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(OUT_DIR)
sys.path.insert(0, PARENT)

import measure_d4_tail_slab as S      # noqa: E402
import measure_d4_tail_vs_R as T      # noqa: E402

import numpy as np                    # noqa: E402

NPZ_IN = os.path.join(PARENT, "d4_tail_slab.npz")
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}


def out_path(name, force, subdir=None):
    """Write only inside this folder (or a subfolder of it), never clobber silently."""
    base = os.path.join(OUT_DIR, subdir) if subdir else OUT_DIR
    os.makedirs(base, exist_ok=True)
    p = os.path.realpath(os.path.join(base, name))
    if not p.startswith(os.path.realpath(OUT_DIR) + os.sep):
        raise RuntimeError(f"refusing to write outside {OUT_DIR}: {p}")
    if os.path.exists(p) and not force:
        raise RuntimeError(f"{p} exists; pass --force to overwrite")
    return p


def run_ladder(atoms_idx, radii, workers, ckpt, every):
    """Every (atom, radius) ball, biggest first, checkpointed and resumable."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    P, _, tree = T._BIG["frame"]
    na, nr = len(atoms_idx), len(radii)

    e2 = np.full((na, nr), np.nan)
    e3 = np.full((na, nr), np.nan)
    nb = np.zeros((na, nr), dtype=int)
    done = np.zeros((na, nr), dtype=bool)

    if os.path.exists(ckpt):
        c = np.load(ckpt)
        if (np.array_equal(c["atoms_idx"], atoms_idx)
                and np.allclose(c["radii"], radii)):
            e2, e3, nb, done = c["e2"], c["e3"], c["nb"], c["done"]
            print(f"  resumed from checkpoint: {done.sum()}/{done.size} balls already done")
        else:
            raise RuntimeError(f"{ckpt} belongs to a different atom set or radius "
                               f"ladder; move it away to start fresh")

    def save():
        tmp = ckpt + ".tmp.npz"
        np.savez(tmp, atoms_idx=atoms_idx, radii=radii,
                 e2=e2, e3=e3, nb=nb, done=done)
        os.replace(tmp, ckpt)

    todo = [(a, k) for a in range(na) for k in range(nr) if not done[a, k]]
    # Largest balls first (longest-processing-time-first), as in the sibling.
    sizes = np.array([len(tree.query_ball_point(P[atoms_idx[a]], radii[k]))
                      for a, k in todo])
    todo = [todo[j] for j in np.argsort(-sizes)]
    print(f"  {len(todo)} balls to run (largest {sizes.max() if len(sizes) else 0} "
          f"atoms, mean {sizes.mean() if len(sizes) else 0:.0f}), {workers} workers")

    t0 = last = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=S._fork_ctx()) as ex:
        futs = {ex.submit(T._ladder_task, (int(atoms_idx[a]), float(radii[k]))): (a, k)
                for a, k in todo}
        for f in as_completed(futs):
            a, k = futs[f]
            e2[a, k], e3[a, k], nb[a, k] = f.result()
            done[a, k] = True
            n_done += 1
            now = time.time()
            if now - last > every:
                save()
                last = now
                rate = n_done / (now - t0)
                left = (len(todo) - n_done) / rate if rate else float("nan")
                print(f"  [{time.strftime('%H:%M:%S')}] {n_done}/{len(todo)} balls, "
                      f"checkpoint written; ~{left/3600:.1f} h left at current rate "
                      f"(big balls run first, so this overestimates)", flush=True)
    save()
    print(f"  finished {len(todo)} balls in {time.time()-t0:.0f}s wall")
    return e2, e3, nb


def build_groups(tail, reg, Z):
    """Mean tail per region x element: {name: (n_atoms, curve over R)}."""
    el = np.array([SYMBOL.get(int(z), str(z)) for z in Z])
    groups = {}
    for r in S.REGIONS:
        for e in sorted(set(el[reg == r])):
            m = (reg == r) & (el == e)
            groups[f"{r.replace('_', ' ')} {e}"] = (int(m.sum()), tail[m].mean(axis=0))
    return groups


def replot(sub):
    """Redraw the plots from a finished run's saved npz. No D4 is run, and
    summary.txt / checkpoint.npz are left untouched."""
    src = os.path.join(OUT_DIR, sub, "d4_tail_vs_R_all_atoms.npz") if sub else \
        os.path.join(OUT_DIR, "d4_tail_vs_R_all_atoms.npz")
    if not os.path.exists(src):
        sys.exit(f"no finished results to replot: {src}")
    a = np.load(src, allow_pickle=True)
    groups = build_groups(a["tail"], a["region"].astype(str), a["Z"])
    for q in make_plots(a["radii"], a["mean_sys"], groups, sub, True, int(a["frame_idx"])):
        print(f"  wrote {os.path.relpath(q, OUT_DIR)}")


def make_plots(radii, mean_sys, groups, outdir_name, force, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mark(ax):
        # No line at D4's 21.17 Ang three-body cutoff: removed at the user's
        # request (nothing visible happens to the curve there).
        ax.axvline(S.RECEPTIVE_FIELD, color="#D55E00", lw=1.6, ls="--", zorder=1)
        ax.axvline(S.D4_CUT_TWOBODY, color="#666666", lw=1.0, ls="-.", zorder=1)

    def label_lines(ax):
        for x, txt, dx in ((S.RECEPTIVE_FIELD, "production 10 $\\AA$", 0.25),
                           (S.D4_CUT_TWOBODY, "D4 2-body 31.75 $\\AA$", -0.25)):
            ax.text(x + dx, 0.97, txt, transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color="#444444", rotation=90,
                    ha="left" if dx > 0 else "right", va="top")

    # ---- 1. the whole-system curve ------------------------------------------
    # Normal-scale panel only. The log-scale panel that used to sit below it
    # is documented in removed_log_panel.md.
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(radii, mean_sys, "-o", ms=4, lw=2.4, color="#111111")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    mark(ax)
    label_lines(ax)
    ax.set_ylabel(r"$\langle E_i^{\rm missing}(R)\rangle$ over all atoms  (meV/atom)")
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_title(f"Whole-system dispersion beyond a cutoff $R$ (every atom, frame {frame})")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p1 = out_path("d4_tail_vs_R_all_atoms.png", force, outdir_name)
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---- 2. breakdown by region x element -----------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = plt.get_cmap("tab10")
    for j, (name, (n, y)) in enumerate(groups.items()):
        ax.plot(radii, y, "-o", ms=2.5, lw=1.2, color=cmap(j % 10),
                label=f"{name} (n={n})")
    ax.plot(radii, mean_sys, "-", lw=2.6, color="#111111", label="whole system", zorder=5)
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    mark(ax)
    label_lines(ax)
    ax.set_ylabel(r"$\langle E_i^{\rm missing}(R)\rangle$  (meV/atom)")
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_title(f"Breakdown by region and element (every atom, frame {frame})")
    ax.legend(fontsize=7, ncol=2, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p2 = out_path("d4_tail_vs_R_by_region_element.png", force, outdir_name)
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    return p1, p2


def main():
    S.ensure_env()
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=float, default=600.0,
                    help="seconds between checkpoint writes")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="3 atoms x 3 radii, written to test_run/, to check the pipeline")
    ap.add_argument("--frame-slot", type=int, default=0,
                    help="which of the 10 frames stored in d4_tail_slab.npz (0 = frame 200). "
                         "Slot 0 writes here; other slots write to frame_NNNN/")
    ap.add_argument("--replot", action="store_true",
                    help="only redraw the plots from this frame's saved npz (no D4)")
    args = ap.parse_args()

    workers = args.workers or int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or os.cpu_count()
    slot = args.frame_slot
    d = np.load(S.guard_read(NPZ_IN), allow_pickle=True)
    if not 0 <= slot < len(d["frame_idx"]):
        sys.exit(f"--frame-slot must be 0..{len(d['frame_idx'])-1}")
    sub = "test_run" if args.test else (None if slot == 0 else f"frame_{int(d['frame_idx'][slot]):04d}")

    if args.replot:
        replot(sub)
        return

    tee = None
    if args.save or args.test:
        tee = S.Tee(sys.stdout, out_path("summary.txt", args.force or args.test, sub))
        sys.stdout = tee

    try:
        from ase.io import read
        from scipy.spatial import cKDTree
        t_start = time.time()
        print("=" * 78)
        print("WHOLE-SYSTEM D4 TAIL vs CUTOFF RADIUS -- every atom, one frame")
        print("=" * 78)

        e_ref = d["e_ref"][slot] * 1000.0                  # meV, this frame
        Z = d["Z"]
        frame0 = int(d["frame_idx"][slot])
        traj = str(d["traj"])
        print(f"  reference : exact periodic E_i^ref reused from d4_tail_slab.npz")
        print(f"  frame     : {frame0} (slot {slot}) of {traj}")

        atoms = read(S.guard_read(traj), index=str(frame0))
        if len(atoms) != len(Z) or not np.array_equal(atoms.numbers, Z):
            raise RuntimeError("npz and trajectory frame disagree")
        # Regions from THIS frame's positions (the npz stores slot 0's only; water
        # moves, ~12% of atoms change region by frame 1520). Must precede wrap(),
        # as in the sibling script.
        region, _, _ = S.assign_regions(atoms, topology=read(S.guard_read(traj), index=0))
        region = np.asarray(region).astype(str)
        if slot == 0 and not np.array_equal(region, d["region"].astype(str)):
            raise RuntimeError("slot 0 regions do not reproduce the npz")
        atoms.wrap()

        radii = T.RADII
        atoms_idx = np.arange(len(atoms))
        if args.test:
            atoms_idx = np.array([int(np.where(region == r)[0][0])
                                  for r in ("graphene", "bulk_water", "water_surface")])
            radii = np.array([10.0, 20.0, S.D4_CUT_TWOBODY])
        print(f"  atoms     : {len(atoms_idx)}   radii: {len(radii)} "
              f"({radii[0]:.2f} .. {radii[-1]:.4f} A)   workers: {workers}")

        P, Zi, nk = S.build_images(atoms, float(T.RADII[-1]) + 2.0)
        T._BIG["frame"] = (P, Zi, cKDTree(P))
        print(f"  images nk={nk}, {len(P)} rows")

        print("\n--- balls ---")
        ckpt = out_path("checkpoint.npz", True, sub)
        e2, e3, nb = run_ladder(atoms_idx, radii, workers, ckpt, args.checkpoint_every)

        e_ball = (e2 + e3) * 1000.0                          # meV
        tail = e_ref[atoms_idx][:, None] - e_ball            # (natoms, nR)
        mean_sys = tail.mean(axis=0)
        base = mean_sys[0]

        # ---- C1 -------------------------------------------------------------
        print("\n--- C1 whole-system tail must reach zero at D4's two-body cutoff ---")
        ratio = abs(mean_sys[-1]) / abs(base)
        print(f"  <E_tail({radii[-1]:.2f})> = {mean_sys[-1]:+.5f} meV/atom  "
              f"vs <E_tail({radii[0]:.1f})> = {base:+.5f}")
        print(f"  residual is {100*ratio:.2f}% of the 10 A tail  "
              f"{'[PASS]' if ratio < 0.05 else '[WARN]'}")

        # ---- C2 -------------------------------------------------------------
        print("\n--- C2 R = 10 must reproduce d4_tail_slab.npz atom by atom ---")
        stored = (d["e_ref"][slot][atoms_idx] - d["e_ball"][slot][atoms_idx]) * 1000.0
        dev = np.abs(tail[:, 0] - stored)
        print(f"  stored <E_tail(10)> = {stored.mean():+.5f}   here = {tail[:, 0].mean():+.5f} meV/atom")
        print(f"  max |deviation| over {len(atoms_idx)} atoms = {dev.max():.6f} meV "
              f"{'[PASS]' if dev.max() < 0.01 else '[WARN]'}")

        # ---- C3 -------------------------------------------------------------
        print("\n--- C3 monotone decay of the whole-system curve ---")
        worst = float(np.max(np.diff(np.abs(mean_sys))))
        print(f"  largest increase in |<E_tail>| between adjacent radii: {worst:+.2e} meV")
        print(f"  {'[PASS]' if worst <= 1e-9 else '[WARN -- non-monotone]'}")

        # ---- groups ---------------------------------------------------------
        reg = region[atoms_idx]
        groups = build_groups(tail, reg, Z[atoms_idx])

        # ---- tables ---------------------------------------------------------
        print("\n--- whole-system <E_tail(R)> (meV/atom), every atom ---")
        print(f"  {'R (A)':>8s} {'<E_tail>':>10s} {'% of 10A':>9s} {'ball':>7s}")
        for k, R in enumerate(radii):
            print(f"  {R:8.4f} {mean_sys[k]:10.4f} {100*mean_sys[k]/base:8.1f}% "
                  f"{nb[:, k].mean():7.0f}")

        print("\n--- by region (meV/atom, all atoms of the region) ---")
        present = [r for r in S.REGIONS if (reg == r).any()]
        print(f"  {'R (A)':>8s}" + "".join(f" {r.replace('_', ' ')[:12]:>13s}" for r in present))
        print(f"  {'n':>8s}" + "".join(f" {int((reg == r).sum()):13d}" for r in present))
        for k, R in enumerate(radii):
            print(f"  {R:8.4f}" + "".join(f" {tail[reg == r, k].mean():13.4f}" for r in present))

        print("\n--- by region x element (meV/atom) ---")
        names = list(groups)
        print(f"  {'R (A)':>8s}" + "".join(f" {n[:14]:>15s}" for n in names))
        print(f"  {'n':>8s}" + "".join(f" {groups[n][0]:15d}" for n in names))
        for k, R in enumerate(radii):
            print(f"  {R:8.4f}" + "".join(f" {groups[n][1][k]:15.4f}" for n in names))

        print("\n--- what a larger receptive field would buy (whole system) ---")
        for R in (12.0, 14.0, 16.0, 20.0, 24.0):
            k = int(np.argmin(np.abs(radii - R)))
            if abs(radii[k] - R) > 1e-6:
                continue
            rec = abs(base) - abs(mean_sys[k])
            print(f"  10.0 -> {R:4.1f} A recovers {rec:7.4f} meV/atom "
                  f"({100*rec/abs(base):5.1f}% of the 10 A tail), leaving {mean_sys[k]:+7.4f}")
        print("\n  NOTE: the MEAN tail is largely a constant that MACE absorbs into its")
        print("  E0s/scale-shift; it answers 'how far does dispersion reach', not")
        print("  'is the model wrong'. See print_verdict() in measure_d4_tail_slab.py.")

        # ---- outputs --------------------------------------------------------
        force = args.force or args.test
        p = out_path("d4_tail_vs_R_all_atoms.npz", force, sub)
        np.savez_compressed(p, radii=radii, atoms_idx=atoms_idx, region=reg, Z=Z[atoms_idx],
                            tail=tail, e2=e2, e3=e3, e_ref=e_ref[atoms_idx], nball=nb,
                            mean_sys=mean_sys, frame_idx=frame0, traj=traj)
        print(f"\n  wrote {os.path.basename(p)}")
        for q in make_plots(radii, mean_sys, groups, sub, force, frame0):
            print(f"  wrote {os.path.basename(q)}")
        print(f"\ntotal wall time {time.time()-t_start:.0f}s")
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


if __name__ == "__main__":
    main()
