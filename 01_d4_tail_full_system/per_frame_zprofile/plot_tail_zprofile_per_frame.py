#!/usr/bin/env python3
"""
The energy panel of d4_tail_zprofile.png, redrawn once per frame.

WHY
---
measure_d4_tail_slab.py averages each atom's tail over the 10 frames and then
bins that average by the atom's height in the FIRST frame only (and assigns its
region from the first frame only). Water moves between frames, so an atom's
later-frame values get credited to the slab it started in, which blurs the
profile wherever it varies with height (surface, interface).

Here nothing is averaged over frames. For each frame k separately:

    E_i^tail = E_i^ref(periodic) - E_i^ball(10 A)     from d4_tail_slab.npz
    z_i, region_i                                      from frame k itself

then atoms are binned in 2 A slabs of height, exactly as in the original plot.

No D4 is recomputed: the per-frame, per-atom energies are already stored in
d4_tail_slab.npz. Only the positions are re-read from the trajectory, because
the npz stores z_rel for the first frame only.

All frames share one bin grid and one y-range so the plots are comparable.

VALIDATION
----------
  P1  frame 0: z_rel and region recomputed here must equal those stored in the
      npz (same frame, same code), otherwise the frames are misaligned.
  P2  every frame must have the same atom count and element order as the npz.

USAGE
    python3 plot_tail_zprofile_per_frame.py [--force]
"""
import os
import sys
import argparse

import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(OUT_DIR)
sys.path.insert(0, PARENT)

import measure_d4_tail_slab as S  # noqa: E402  (region code is reused, not copied)

NPZ_IN = os.path.join(PARENT, "d4_tail_slab.npz")
BIN = 2.0  # A, same as the original plot

COLORS = {"graphene": "#444444", "adsorbate": "#0072B2",
          "interfacial_water": "#D55E00", "bulk_water": "#009E73",
          "water_surface": "#CC79A7"}


def out_path(name, force):
    """Write only inside this folder, and never clobber silently."""
    p = os.path.realpath(os.path.join(OUT_DIR, name))
    if not p.startswith(os.path.realpath(OUT_DIR) + os.sep):
        raise RuntimeError(f"refusing to write outside {OUT_DIR}: {p}")
    if os.path.exists(p) and not force:
        raise RuntimeError(f"{p} exists; pass --force to overwrite")
    return p


def profile(z_rel, e_tail, bins):
    """Mean tail and atom count per slab; NaN where a slab is empty."""
    which = np.digitize(z_rel, bins) - 1
    nb = len(bins) - 1
    mean = np.full(nb, np.nan)
    count = np.zeros(nb, dtype=int)
    for k in range(nb):
        m = which == k
        count[k] = m.sum()
        if count[k]:
            mean[k] = e_tail[m].mean()
    return mean, count


def draw(ax, centers, prof, z_rel, region, title):
    """Same look as the top panel of d4_tail_zprofile.png."""
    import matplotlib.pyplot as plt
    ax.plot(centers, prof, "-o", ms=3, color="#333333")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    for r in S.REGIONS:
        m = region == r
        if m.any():
            # Graphene is a flat sheet (zero width); give every band a floor.
            lo, hi = z_rel[m].min(), z_rel[m].max()
            if hi - lo < 0.6:
                lo, hi = lo - 0.3, hi + 0.3
            ax.axvspan(lo, hi, alpha=0.10, color=COLORS[r], lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[r], alpha=0.35)
               for r in S.REGIONS if (region == r).any()]
    labels = [r.replace("_", " ") for r in S.REGIONS if (region == r).any()]
    ax.legend(handles, labels, fontsize=8, loc="lower right", frameon=False)
    ax.set_ylabel(r"$\langle E_i^{\rm tail}\rangle$  (meV)")
    ax.set_xlabel(r"$z$ relative to the graphene plane ($\AA$)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.io import Trajectory

    d = np.load(S.guard_read(NPZ_IN), allow_pickle=True)
    fidx = d["frame_idx"]
    e_tail = (d["e_ref"] - d["e_ball"]) * 1000.0          # (nframes, natoms), meV
    traj = str(d["traj"])
    tr = Trajectory(S.guard_read(traj))
    topo = tr[0]  # same topology source as the original run

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 78)
    log("D4 TAIL z-PROFILE, ONE PLOT PER FRAME (no averaging over frames)")
    log("=" * 78)
    log(f"  source npz : {NPZ_IN}")
    log(f"  trajectory : {traj}")
    log(f"  frames     : {list(map(int, fidx))}")

    # ---- per-frame heights and regions --------------------------------------
    z_all, reg_all = [], []
    for k, fi in enumerate(fidx):
        atoms = tr[int(fi)]
        # P2
        if len(atoms) != e_tail.shape[1] or not np.array_equal(atoms.numbers, d["Z"]):
            raise RuntimeError(f"frame {fi}: atoms do not match the npz")
        region, z_rel, _ = S.assign_regions(atoms, topology=topo)
        z_all.append(np.asarray(z_rel))
        reg_all.append(np.asarray(region).astype(str))

    # ---- P1 -----------------------------------------------------------------
    dz = np.abs(z_all[0] - d["z_rel"]).max()
    same_reg = np.array_equal(reg_all[0], d["region"].astype(str))
    log("\n--- P1 frame 0 must reproduce the stored heights and regions ---")
    log(f"  max |z_rel - stored| = {dz:.2e} A,  regions identical: {same_reg}  "
        f"{'[PASS]' if dz < 1e-6 and same_reg else '[FAIL]'}")
    if not (dz < 1e-6 and same_reg):
        raise RuntimeError("frame 0 does not reproduce the npz; frames are misaligned")

    # ---- one bin grid and y-range for every frame ---------------------------
    zmin = min(z.min() for z in z_all)
    zmax = max(z.max() for z in z_all)
    bins = np.arange(np.floor(zmin), np.ceil(zmax) + BIN, BIN)
    centers = 0.5 * (bins[1:] + bins[:-1])

    profs, counts = [], []
    for k in range(len(fidx)):
        p, c = profile(z_all[k], e_tail[k], bins)
        profs.append(p)
        counts.append(c)
    profs = np.array(profs)
    counts = np.array(counts)

    lo = np.nanmin(profs)
    ylim = (lo - 0.05 * abs(lo), 0.08)

    # ---- how many atoms change region relative to frame 0 -------------------
    log("\n--- region sizes per frame (atoms) ---")
    hdr = f"  {'frame':>6s}" + "".join(f" {r.replace('_', ' ')[:12]:>13s}" for r in S.REGIONS)
    hdr += f" {'moved region':>13s}"
    log(hdr)
    for k, fi in enumerate(fidx):
        row = f"  {int(fi):6d}" + "".join(
            f" {int((reg_all[k] == r).sum()):13d}" for r in S.REGIONS)
        row += f" {int((reg_all[k] != reg_all[0]).sum()):13d}"
        log(row)
    log("  (moved region = atoms whose region differs from frame 0)")

    # ---- the profiles as a table --------------------------------------------
    log("\n--- <E_tail> per 2 A slab (meV); columns = frames ---")
    log(f"  {'z centre':>8s}" + "".join(f" {int(fi):>7d}" for fi in fidx)
        + f" {'mean':>7s} {'sd':>6s}")
    for b in range(len(centers)):
        v = profs[:, b]
        if np.all(np.isnan(v)):
            continue
        row = f"  {centers[b]:8.1f}" + "".join(
            f" {x:7.3f}" if np.isfinite(x) else f" {'-':>7s}" for x in v)
        row += f" {np.nanmean(v):7.3f} {np.nanstd(v, ddof=1):6.3f}"
        log(row)

    # ---- one figure per frame -----------------------------------------------
    written = []
    for k, fi in enumerate(fidx):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        draw(ax, centers, profs[k], z_all[k], reg_all[k],
             f"D4 beyond the 10 $\\AA$ receptive field - frame {int(fi)}")
        ax.set_ylim(*ylim)
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)
        fig.tight_layout()
        p = out_path(f"d4_tail_zprofile_frame{int(fi):04d}.png", args.force)
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(os.path.basename(p))

    # ---- all frames on one axis, for comparison -----------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cmap = plt.get_cmap("viridis")
    for k, fi in enumerate(fidx):
        ax.plot(centers, profs[k], "-o", ms=2, lw=1.0, alpha=0.8,
                color=cmap(k / max(len(fidx) - 1, 1)), label=f"frame {int(fi)}")
    ax.axhline(0, lw=0.8, color="k", alpha=0.4)
    ax.set_ylim(*ylim)
    ax.set_xlim(bins[0] - 1, bins[-1] + 1)
    ax.set_ylabel(r"$\langle E_i^{\rm tail}\rangle$  (meV)")
    ax.set_xlabel(r"$z$ relative to the graphene plane ($\AA$)")
    ax.set_title("All 10 frames (each binned by its own heights)")
    ax.legend(fontsize=7, ncol=2, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p = out_path("d4_tail_zprofile_all_frames.png", args.force)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(os.path.basename(p))

    p = out_path("d4_tail_zprofile_per_frame.npz", args.force)
    np.savez_compressed(p, frame_idx=fidx, bins=bins, centers=centers,
                        profiles=profs, counts=counts,
                        z_rel=np.array(z_all), region=np.array(reg_all))
    written.append(os.path.basename(p))

    log("\n--- wrote ---")
    for w in written:
        log(f"  {w}")

    with open(out_path("summary.txt", args.force), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
