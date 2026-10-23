#!/usr/bin/env python3
"""
Does the D4 energy beyond 10 Ang change the RANKING of the 26 adsorbate-substrate
structures?

No D4 is run here. This reads two existing results:

  d4_tail_slab_summary.txt    REGISTRY MODULE of job 1397668 (2026-09-18):
                              per structure, the part of E_ads a 10 Ang model
                              misses,  tail = t(system) - t(substrate) - t(adsorbate),
                              with t = periodic D4 - D4 inside 10 Ang balls
  ads_eng_calc/DFT/adsorption_energies.csv
                              DFT E_ads (full D4) per structure

A 10 Ang model sees  E_ads(10 A) = E_ads(DFT) - tail.  The question is whether
ordering the structures by E_ads(10 A) instead of E_ads(DFT) swaps any pair.

USAGE
    python3 check_eads_ranking.py            # writes eads_ranking/
"""
import os
import re
import csv
import itertools

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(THIS_DIR, "d4_tail_slab_summary.txt")
ADS_CSV = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT/adsorption_energies.csv"
OUT = os.path.join(THIS_DIR, "eads_ranking")


def read_tails():
    txt = open(SUMMARY).read()
    block = txt.split("E_ads tail contribution (meV)")[1].split("  mean")[0]
    tails = {m.group(1): float(m.group(2))
             for m in re.finditer(r"^\s+(m\S+)\s+(-?\d+\.\d+)\s*$", block, re.M)}
    mol = float(re.search(r"isolated adsorbate tail:\s*(-?\d+\.\d+)", txt).group(1))
    return tails, mol


def main():
    os.makedirs(OUT, exist_ok=True)
    tails, t_mol = read_tails()
    dft = {r["label"]: 1000.0 * float(r["e_ads"]) for r in csv.DictReader(open(ADS_CSV))}
    tags = sorted(set(tails) & set(dft), key=lambda t: dft[t])
    if len(tags) != len(tails):
        raise RuntimeError(f"structures without a DFT E_ads: {sorted(set(tails) - set(dft))}")

    e_full = np.array([dft[t] for t in tags])            # meV, full D4
    tail = np.array([tails[t] for t in tags])            # meV, missed at 10 Ang
    e_10 = e_full - tail                                 # what a 10 Ang model sees
    rank_full = np.argsort(np.argsort(e_full)) + 1
    rank_10 = np.argsort(np.argsort(e_10)) + 1

    pairs = list(itertools.combinations(range(len(tags)), 2))
    flips = [(i, j) for i, j in pairs
             if np.sign(e_full[i] - e_full[j]) != np.sign(e_10[i] - e_10[j])]
    # Pairs whose DFT gap is smaller than their tail difference: these could
    # have flipped, and did not only because the difference had the right sign.
    at_risk = [(i, j) for i, j in pairs
               if abs(e_full[i] - e_full[j]) < abs(tail[i] - tail[j])]

    # ---- CSV --------------------------------------------------------------
    p_csv = os.path.join(OUT, "eads_tail_10A.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["structure", "e_ads_full_d4_meV", "tail_beyond_10A_meV",
                    "e_ads_10A_model_meV", "rank_full_d4", "rank_10A"])
        for k, t in enumerate(tags):
            w.writerow([t, f"{e_full[k]:.3f}", f"{tail[k]:.3f}", f"{e_10[k]:.3f}",
                        rank_full[k], rank_10[k]])

    # ---- summary ----------------------------------------------------------
    lines = [
        "E_ads RANKING WITH AND WITHOUT THE D4 BEYOND 10 A",
        "=" * 60,
        f"source of tails : {SUMMARY} (job 1397668, 2026-09-18)",
        f"source of E_ads : {ADS_CSV}",
        f"structures      : {len(tags)}",
        f"isolated adsorbate's own tail: {t_mol:.3f} meV (already inside each value)",
        "",
        "tail = part of E_ads a 10 A model misses (meV)",
        f"  mean {tail.mean():.3f}   sd {tail.std(ddof=1):.3f}   "
        f"min {tail.min():.3f}   max {tail.max():.3f}   max-min {np.ptp(tail):.3f}",
        f"  relative to mean |E_ads| {abs(e_full.mean()):.0f} meV: "
        f"{100*abs(tail.mean())/abs(e_full.mean()):.2f}%",
        "",
        f"DFT E_ads spread across structures: {np.ptp(e_full):.1f} meV",
        f"pairs compared        : {len(pairs)}",
        f"pairs that swap order : {len(flips)}",
        f"pairs whose DFT gap < their tail difference (could have swapped): {len(at_risk)}",
    ]
    for i, j in at_risk:
        lines.append(f"    {tags[i]} vs {tags[j]}: DFT gap {e_full[i]-e_full[j]:+.3f}, "
                     f"tail difference {tail[i]-tail[j]:+.3f} meV")
    lines += ["", f"ranks identical: {bool(np.array_equal(rank_full, rank_10))}", "",
              f"{'rank':>4s}  {'structure':<22s} {'E_ads full D4':>14s} {'tail':>9s} "
              f"{'E_ads 10 A':>11s} {'rank 10 A':>9s}"]
    for k, t in enumerate(tags):
        lines.append(f"{rank_full[k]:4d}  {t:<22s} {e_full[k]:14.1f} {tail[k]:9.3f} "
                     f"{e_10[k]:11.1f} {rank_10[k]:9d}")
    p_txt = os.path.join(OUT, "summary.txt")
    open(p_txt, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, len(tags) + 1)

    # 1. the tail per structure: flat
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x, tail, "o", ms=6, color="#0072B2")
    ax.axhline(tail.mean(), color="#111111", lw=1.2, ls="--")
    ax.text(len(tags) + 0.4, tail.mean(), f"mean {tail.mean():.1f} meV", va="bottom",
            ha="right", fontsize=8.5, color="#111111")
    ax.set_ylim(tail.mean() - 5, tail.mean() + 5)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=90, fontsize=6.5)
    ax.set_xlabel("structure (ordered by DFT $E_{\\rm ads}$, most stable left)")
    ax.set_ylabel("part of $E_{\\rm ads}$ beyond 10 $\\AA$  (meV)")
    ax.set_title(f"D4 beyond 10 $\\AA$ shifts every $E_{{\\rm ads}}$ by ~{abs(tail.mean()):.0f} meV "
                 f"(spread {np.ptp(tail):.1f} meV)")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p1 = os.path.join(OUT, "eads_tail_per_structure.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # 2. relative E_ads, full D4 vs 10 A model: same order
    rel_full = e_full - e_full.min()
    rel_10 = e_10 - e_10.min()
    main = rel_full < 200.0                     # the two ~1.5 eV outliers go in the note
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x[main], rel_full[main], "o", ms=7, color="#0072B2", label="full D4 (DFT)")
    ax.plot(x[main], rel_10[main], "o", ms=11, mfc="none", mew=1.4, color="#E69F00",
            label="D4 cut at 10 $\\AA$")
    ax.set_xticks(x[main])
    ax.set_xticklabels(np.array(tags)[main], rotation=90, fontsize=6.5)
    ax.set_xlabel("structure (ordered by DFT $E_{\\rm ads}$)")
    ax.set_ylabel("$E_{\\rm ads}$ relative to the most stable  (meV)")
    ax.set_title(f"Ranking unchanged: {len(flips)} of {len(pairs)} pairs swap order")
    off = [f"{tags[k]}: +{rel_full[k]:.0f} meV" for k in np.where(~main)[0]]
    if off:
        ax.text(0.02, 0.97, "off scale:\n" + "\n".join(off), transform=ax.transAxes,
                fontsize=7.5, va="top", color="#444444")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p2 = os.path.join(OUT, "eads_ranking_full_vs_10A.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    for p in (p_csv, p_txt, p1, p2):
        print(f"wrote {os.path.relpath(p, THIS_DIR)}")


if __name__ == "__main__":
    main()
