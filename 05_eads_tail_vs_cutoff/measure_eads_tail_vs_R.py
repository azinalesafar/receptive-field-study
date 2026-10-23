#!/usr/bin/env python3
"""
Does increasing the cutoff from 10 Ang change the RANKING of the 26
adsorbate-substrate structures? E_ads missing energy as a function of R.

WHAT THIS ADDS TO registry_module() IN ../measure_d4_tail_slab.py
-----------------------------------------------------------------
That module (job 1397668, 2026-09-18) computed, at R = 10 Ang only, the part of
E_ads a 10 Ang model misses,

    tail_ads(R) = t_system(R) - t_substrate(R) - t_adsorbate(R),
    t_X(R)      = sum_i [ E_i^ref(periodic) - E_i^ball(R) ]   over all atoms of X,

and found it ~ -97.7 meV for every structure (sd 0.38): no ranking change.
Here the same quantity is computed on the full radius ladder of
measure_d4_tail_vs_R.py (10 -> 31.75 Ang), so the ranking can be checked at
any cutoff a larger model might use:

    E_ads(R) = E_ads(DFT, full D4) - tail_ads(R)

and each E_ads(R) ranking is compared with the DFT ranking, pair by pair.
The three per-structure terms are saved separately this time.

PERIODIC IMAGES
---------------
Systems and substrates are periodic in x, y only (as in the files); the balls
are built from as many image cells as needed to fill 31.75 Ang, exactly as the
periodic D4 reference counts them, so a 2x2 supercell would give the same
numbers (checked on the frame-1520 slab: identical to 6 digits).

The isolated adsorbate file is periodic in x, y AND z (35.6 x 36.1 x 19.8 Ang),
so its D4 includes its own images 19.8 Ang away in z. That term is the same for
all 26 structures, so it cannot change the ranking; it only moves the absolute
offset. It is computed both as in the file ("mol", used for E_ads, consistent
with the DFT reference built from that file) and fully non-periodic
("mol_iso", reported for comparison).

VALIDATION BUILT IN
-------------------
  C1  tail_ads(31.75) ~ 0 for every structure.
  C2  tail_ads(10) reproduces the job-1397668 values in d4_tail_slab_summary.txt.

USAGE
    python3 measure_eads_tail_vs_R.py --workers 128 --save
    python3 measure_eads_tail_vs_R.py --test          # 2 structures, R = 10 only
    python3 measure_eads_tail_vs_R.py --replot
"""
import os
import re
import sys
import csv
import glob
import time
import argparse
import itertools

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(OUT_DIR)
sys.path.insert(0, PARENT)

import measure_d4_tail_slab as S     # noqa: E402
import measure_d4_tail_vs_R as T     # noqa: E402

import numpy as np                   # noqa: E402

SEPT_SUMMARY = os.path.join(PARENT, "d4_tail_slab_summary.txt")
OUT_NPZ = "eads_tail_vs_R.npz"

# Structure payloads live in a module global so forked workers inherit them
# (same pattern, and same fork requirement, as S._FRAMES / T._BIG).
_FR = {}


def out_path(name, force):
    p = os.path.realpath(os.path.join(OUT_DIR, name))
    if not p.startswith(os.path.realpath(OUT_DIR) + os.sep):
        raise RuntimeError(f"refusing to write outside {OUT_DIR}: {p}")
    if os.path.exists(p) and not force:
        raise RuntimeError(f"{p} exists; pass --force to overwrite")
    return p


def _task(args):
    """One structure, one atom, one radius -> energy of that atom in the ball (eV).
    Same estimator as T._ladder_task, but for any registered structure."""
    key, i, R = args
    P, Z, tree = _FR[key]
    sel = np.asarray(sorted(tree.query_ball_point(P[i], R)))
    loc = int(np.searchsorted(sel, i))
    m = S._model(Z[sel], P[sel])
    pw = m.get_pairwise_dispersion(S._param())
    e = (float(np.asarray(pw["additive pairwise energy"])[loc].sum())
         + float(np.asarray(pw["non-additive pairwise energy"])[loc].sum())) * S.HARTREE
    return e


def load_structures():
    from ase.io import read
    subs = {os.path.basename(os.path.dirname(p)): p for p in sorted(glob.glob(S.SUBSTRATE_GLOB))}
    syss = {os.path.basename(os.path.dirname(p)): p for p in sorted(glob.glob(S.SYSTEM_GLOB))}
    tags = sorted(set(subs) & set(syss))
    structs = {}
    for t in tags:
        structs[f"sys:{t}"] = read(S.guard_read(syss[t]))
        structs[f"sub:{t}"] = read(S.guard_read(subs[t]))
    mol = read(S.guard_read(S.ISOLATED_ADS))
    structs["mol"] = mol
    iso = mol.copy()
    iso.pbc = False
    structs["mol_iso"] = iso
    return tags, structs


def sept_values():
    txt = open(SEPT_SUMMARY).read()
    block = txt.split("E_ads tail contribution (meV)")[1].split("  mean")[0]
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"^\s+(m\S+)\s+(-?\d+\.\d+)\s*$", block, re.M)}


def dft_eads():
    return {r["label"]: 1000.0 * float(r["e_ads"]) for r in csv.DictReader(open(S.ADS_CSV))}


def ranking_stats(e_full, tail):
    """Pairs whose order under E_full - tail differs from the order under E_full."""
    e_R = e_full - tail
    pairs = list(itertools.combinations(range(len(e_full)), 2))
    flips = [(i, j) for i, j in pairs
             if np.sign(e_full[i] - e_full[j]) != np.sign(e_R[i] - e_R[j])]
    risk = [(i, j) for i, j in pairs
            if abs(e_full[i] - e_full[j]) < abs(tail[i] - tail[j])]
    return len(pairs), flips, risk


def run(tasks, workers, ckpt, every):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n = len(tasks)
    keys = np.array([f"{k}|{i}|{R:.4f}" for k, i, R in tasks])
    e = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    if os.path.exists(ckpt):
        c = np.load(ckpt)
        if not np.array_equal(c["keys"], keys):
            raise RuntimeError(f"{ckpt} belongs to a different task list; move it away")
        e, done = c["e"], c["done"]
        print(f"  resumed from checkpoint: {done.sum()}/{n} balls already done")

    def save():
        tmp = ckpt + ".tmp.npz"
        np.savez(tmp, keys=keys, e=e, done=done)
        os.replace(tmp, ckpt)

    todo = np.where(~done)[0]
    sizes = np.array([len(_FR[tasks[t][0]][2].query_ball_point(
        _FR[tasks[t][0]][0][tasks[t][1]], tasks[t][2])) for t in todo])
    todo = todo[np.argsort(-sizes)]                   # biggest balls first
    print(f"  {len(todo)} balls to run (largest {sizes.max() if len(sizes) else 0} atoms, "
          f"mean {sizes.mean() if len(sizes) else 0:.0f}), {workers} workers", flush=True)

    t0 = last = time.time()
    k = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=S._fork_ctx()) as ex:
        futs = {ex.submit(_task, tasks[t]): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            e[t] = f.result()
            done[t] = True
            k += 1
            now = time.time()
            if now - last > every:
                save()
                last = now
                print(f"  [{time.strftime('%H:%M:%S')}] {k}/{len(todo)} balls, checkpoint "
                      f"written; ~{(len(todo)-k)/(k/(now-t0))/3600:.1f} h left "
                      f"(big balls first, so this overestimates)", flush=True)
    save()
    print(f"  finished {len(todo)} balls in {time.time()-t0:.0f}s wall")
    return e


def make_plots(radii, tags, tail, e_full, flips_by_R, force):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mark(ax):
        ax.axvline(S.RECEPTIVE_FIELD, color="#D55E00", lw=1.6, ls="--", zorder=1)
        ax.axvline(S.D4_CUT_TWOBODY, color="#666666", lw=1.0, ls="-.", zorder=1)
        for x, txt, dx in ((S.RECEPTIVE_FIELD, "production 10 $\\AA$", 0.25),
                           (S.D4_CUT_TWOBODY, "D4 2-body 31.75 $\\AA$", -0.25)):
            ax.text(x + dx, 0.97, txt, transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color="#444444", rotation=90,
                    ha="left" if dx > 0 else "right", va="top")

    out = []

    # 1. every structure's missing E_ads vs R (the Fig. 6 analogue)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for k in range(len(tags)):
        ax.plot(radii, tail[k], "-", lw=0.9, color="#0072B2", alpha=0.45)
    ax.plot(radii, tail.mean(axis=0), "-o", ms=3.5, lw=2.4, color="#111111",
            label=f"mean of {len(tags)} structures")
    ax.plot([], [], "-", lw=0.9, color="#0072B2", alpha=0.6, label="each structure")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    mark(ax)
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_ylabel(r"part of $E_{\rm ads}$ beyond $R$  (meV)")
    ax.set_title(r"Adsorption energy missed by a model with cutoff $R$ (26 structures)")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p = out_path("eads_tail_vs_R.png", force)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)

    # 2. the differences between structures vs R, with pairs swapped
    spread = tail.max(axis=0) - tail.min(axis=0)
    sd = tail.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(radii, spread, "-o", ms=4, lw=2.2, color="#0072B2", label="largest difference (max - min)")
    ax.plot(radii, sd, "-s", ms=3.5, lw=1.6, color="#E69F00", label="standard deviation")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    mark(ax)
    ax.set_xlabel(r"cutoff radius $R$  ($\AA$)")
    ax.set_ylabel("difference between structures  (meV)")
    nflip = ", ".join(f"{R:g} $\\AA$: {flips_by_R[k]}" for k, R in enumerate(radii)
                      if R in (10.0, 12.0, 15.0, 20.0))
    ax.set_title("How much the missing $E_{\\rm ads}$ differs between structures\n"
                 f"pairs (of 325) swapped vs DFT: {nflip}; max over all $R$: {max(flips_by_R)}",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p = out_path("eads_tail_spread_vs_R.png", force)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)
    return out


def analyse(radii, tags, t_sys, t_sub, t_mol, t_mol_iso, force, save):
    """Print tables, write CSV/npz/plots. All t_* in meV, shape (26, nR) or (nR,)."""
    tail = t_sys - t_sub - t_mol[None, :]
    dft = dft_eads()
    e_full = np.array([dft[t] for t in tags])
    sept = sept_values()
    k10 = int(np.argmin(np.abs(radii - 10.0)))

    print("\n--- C1 every structure's missing E_ads must reach ~0 at 31.75 A ---")
    print(f"  |tail(31.75)| max {np.abs(tail[:, -1]).max():.3f} meV  vs  "
          f"|tail(10)| mean {np.abs(tail[:, k10]).mean():.3f} meV  "
          f"{'[PASS]' if np.abs(tail[:, -1]).max() < 0.05*np.abs(tail[:, k10]).mean() else '[WARN]'}")

    print("\n--- C2 R = 10 must reproduce job 1397668 (d4_tail_slab_summary.txt) ---")
    dev = np.array([tail[k, k10] - sept[t] for k, t in enumerate(tags)])
    print(f"  max |deviation| over {len(tags)} structures: {np.abs(dev).max():.4f} meV "
          f"(Sept. values printed to 3 decimals)  {'[PASS]' if np.abs(dev).max() < 0.002 else '[WARN]'}")

    print("\n--- the isolated adsorbate term (same for all structures) ---")
    print(f"  {'R (A)':>8s} {'as in file (pbc xyz)':>21s} {'non-periodic':>13s} {'difference':>11s}")
    for k, R in enumerate(radii):
        print(f"  {R:8.4f} {t_mol[k]:21.3f} {t_mol_iso[k]:13.3f} {t_mol[k]-t_mol_iso[k]:11.3f}")

    print("\n--- missing E_ads vs cutoff, and whether the ranking changes ---")
    print(f"  {'R (A)':>8s} {'mean':>9s} {'sd':>7s} {'max-min':>8s} {'swapped':>8s} {'at risk':>8s}")
    flips_by_R = []
    for k, R in enumerate(radii):
        npairs, flips, risk = ranking_stats(e_full, tail[:, k])
        flips_by_R.append(len(flips))
        print(f"  {R:8.4f} {tail[:, k].mean():9.3f} {tail[:, k].std(ddof=1):7.3f} "
              f"{np.ptp(tail[:, k]):8.3f} {len(flips):8d} {len(risk):8d}")
    print(f"  ({npairs} pairs; 'swapped' = order differs from DFT; 'at risk' = DFT gap "
          f"smaller than the pair's tail difference)")
    for k, R in enumerate(radii):
        _, flips, _ = ranking_stats(e_full, tail[:, k])
        for i, j in flips:
            print(f"  SWAP at R={R:.2f}: {tags[i]} vs {tags[j]}, DFT gap "
                  f"{e_full[i]-e_full[j]:+.3f}, tail difference {tail[i, k]-tail[j, k]:+.3f} meV")

    print("\n--- per structure (meV), ordered by DFT E_ads ---")
    order = np.argsort(e_full)
    show = [k for k, R in enumerate(radii) if R in (10.0, 12.0, 15.0, 20.0, 24.0)]
    print(f"  {'structure':<20s} {'E_ads DFT':>10s}" + "".join(f" {'R='+format(radii[k],'g'):>9s}" for k in show))
    for o in order:
        print(f"  {tags[o]:<20s} {e_full[o]:10.1f}" + "".join(f" {tail[o, k]:9.3f}" for k in show))

    if not save:
        return
    p = out_path("eads_tail_vs_R.csv", force)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["structure", "R_A", "e_ads_dft_meV", "tail_ads_meV",
                    "t_system_meV", "t_substrate_meV", "t_adsorbate_meV", "t_adsorbate_nonperiodic_meV"])
        for k, t in enumerate(tags):
            for r, R in enumerate(radii):
                w.writerow([t, f"{R:.4f}", f"{e_full[k]:.3f}", f"{tail[k, r]:.4f}",
                            f"{t_sys[k, r]:.4f}", f"{t_sub[k, r]:.4f}",
                            f"{t_mol[r]:.4f}", f"{t_mol_iso[r]:.4f}"])
    print(f"\n  wrote {os.path.basename(p)}")
    for q in make_plots(radii, tags, tail, e_full, flips_by_R, force):
        print(f"  wrote {os.path.basename(q)}")


def main():
    S.ensure_env()
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=float, default=300.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true", help="2 structures at R = 10, printed only")
    ap.add_argument("--replot", action="store_true", help="re-analyse the saved npz (no D4)")
    args = ap.parse_args()

    if args.replot:
        a = np.load(os.path.join(OUT_DIR, OUT_NPZ), allow_pickle=True)
        analyse(a["radii"], list(a["tags"]), a["t_sys"], a["t_sub"], a["t_mol"],
                a["t_mol_iso"], True, True)
        return

    workers = args.workers or int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or os.cpu_count()
    tee = None
    if args.save:
        tee = S.Tee(sys.stdout, out_path("summary.txt", args.force))
        sys.stdout = tee
    try:
        from scipy.spatial import cKDTree
        t_start = time.time()
        print("=" * 78)
        print("E_ads MISSING ENERGY vs CUTOFF -- 26 adsorbate-substrate structures")
        print("=" * 78)
        tags, structs = load_structures()
        radii = T.RADII
        if args.test:
            tags = tags[:2]
            structs = {k: v for k, v in structs.items()
                       if k in ("mol", "mol_iso") or k.split(":", 1)[1] in tags}
            radii = np.array([10.0])       # cheap enough for a login node; checks C2
        print(f"  structures: {len(tags)} systems + {len(tags)} substrates + adsorbate "
              f"(as in file, and non-periodic)")
        print(f"  radii     : {len(radii)} ({radii[0]:.2f} .. {radii[-1]:.4f} A), workers {workers}")

        # ---- periodic references and image frames ----------------------------
        t0 = time.time()
        e_ref = {}
        for key, a in structs.items():
            a = a.copy()
            if a.pbc.any():
                a.wrap()
            e_ref[key] = S.d4_reference(a, want_grad=False)["E_i"] * 1000.0   # meV
            P, Z, _ = S.build_images(a, float(T.RADII[-1]) + 2.0)
            _FR[key] = (P, Z, cKDTree(P))
        print(f"  references: {len(structs)} periodic D4 calls in {time.time()-t0:.0f}s")

        tasks = [(key, i, float(R)) for key in structs for i in range(len(structs[key]))
                 for R in radii]
        print(f"  balls     : {len(tasks)}")

        print("\n--- balls ---")
        if args.test:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers, mp_context=S._fork_ctx()) as ex:
                e = np.array(list(ex.map(_task, tasks, chunksize=4)))
        else:
            e = run(tasks, workers, out_path("checkpoint.npz", True), args.checkpoint_every)
        e *= 1000.0                                                  # meV

        # ---- per-structure missing energy t_X(R) --------------------------------
        t = {}
        pos = 0
        for key, a in structs.items():
            n = len(a)
            ball = e[pos:pos + n * len(radii)].reshape(n, len(radii))
            t[key] = (e_ref[key][:, None] - ball).sum(axis=0)
            pos += n * len(radii)
        t_sys = np.array([t[f"sys:{g}"] for g in tags])
        t_sub = np.array([t[f"sub:{g}"] for g in tags])

        if args.save:
            p = out_path(OUT_NPZ, args.force)
            np.savez_compressed(p, radii=radii, tags=np.array(tags), t_sys=t_sys, t_sub=t_sub,
                                t_mol=t["mol"], t_mol_iso=t["mol_iso"])
            print(f"\n  wrote {os.path.basename(p)}")

        if args.test:
            for g in tags:
                tail = t[f"sys:{g}"] - t[f"sub:{g}"] - t["mol"]
                print(f"  {g}: tail_ads " + "  ".join(f"R={R:.2f}: {tail[k]:+.3f}"
                                                      for k, R in enumerate(radii)) + " meV")
            print(f"  adsorbate term: as in file {t['mol']}, non-periodic {t['mol_iso']}")
            print(f"  Sept. 10 A values: " + ", ".join(f"{g}: {sept_values()[g]}" for g in tags))
            return
        analyse(radii, tags, t_sys, t_sub, t["mol"], t["mol_iso"], args.force, args.save)
        print(f"\ntotal wall time {time.time()-t_start:.0f}s")
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


if __name__ == "__main__":
    main()
