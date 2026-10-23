"""
Automatic parity plot + summary report for the model11 dispersion comparison.

Runs (normally as job_report_m11.slurm, after the model11_no_d4 relaxations finish):
  1. compute_eads.py              -> eads_all_variants.csv (all variants, incl. model11 family)
  2. plot_eads_parity_s18.py m11  -> eads_parity_s18_style_m11.png
  3. writes report_m11/report_model11_dispersion.md with the numbers and a summary

The three series (all vs DFT PBE+D4, same 26 cells, same DFT start geometries):
  m11nod4        no dispersion          model11_no_d4 (PBE-only labels, nothing added)
  m11_d4trained  learned dispersion     model11 (PBE+D4 labels)
  m11nod4_d4     post-hoc dispersion    model11_no_d4 + D4 inside the relaxation
Missing pieces are reported in the file instead of crashing, so the report always appears.
"""
import csv
import datetime
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_m11")
DFT_DIR = "/scratch/johnt447/azin/analysis/ads_eng_calc/DFT"
N_ADS = 126
SMALL = {"m-8_n-6_m-5_n3", "m2_n8_m8_n5"}
SERIES = [  # tag, short name, how dispersion is handled
    ("m11nod4", "no dispersion", "model11_no_d4: fine-tuned on PBE-only labels (D4 stripped), "
     "nothing added at inference"),
    ("m11_d4trained", "learned dispersion", "model11: fine-tuned on PBE+D4 labels, so D4 is learned "
     "implicitly within the 10 Å receptive field (5 Å cutoff × 2 layers)"),
    ("m11nod4_d4", "post-hoc dispersion", "model11_no_d4 + full-range D4 (dftd4, method=PBE, same as "
     "the DFT labels) added to every force call, i.e. during the relaxation"),
]
sys.path.insert(0, HERE)
from plot_eads_parity_s18 import M11_SEEN  # noqa: E402


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "arn" not in l)
    return r.returncode, out


def height(atoms):
    n = len(atoms) - N_ADS
    return atoms.positions[n:, 2].mean() - atoms.positions[:n, 2].max()


def d4(atoms):
    from dftd4.ase import DFTD4
    a = atoms.copy()
    a.calc = DFTD4(method="PBE")
    return a.get_potential_energy()


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    rc1, o1 = run(["compute_eads.py"])
    rc2, o2 = run(["plot_eads_parity_s18.py", "m11"])
    rc3, o3 = run(["plot_eads_per_cell.py"])
    log += [o1, o2, o3]

    e = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "eads_all_variants.csv"))):
        e[r["variant"]][r["label"]] = (float(r["e_ads"]), r["converged"] == "True", int(r["fragments"]))
    for r in csv.DictReader(open(os.path.join(HERE, "dft_references.csv"))):
        e[r["variant"]][r["label"]] = (float(r["e_ads"]), True, 0)
    labels = sorted(e["DFT"], key=lambda l: e["DFT"][l][0])
    x = np.array([e["DFT"][l][0] for l in labels])
    seen = np.array([l in M11_SEEN for l in labels])
    small = np.array([l in SMALL for l in labels])
    h_dft = np.array([height(read(f"{DFT_DIR}/unitcell_candidates/DFT_relaxed/{l}/{l}_system_relaxed.xyz"))
                      for l in labels])

    stats, missing = {}, []
    for tag, name, _ in SERIES:
        if len(e.get(tag, {})) < len(labels):
            missing.append(f"{tag} ({len(e.get(tag, {}))}/{len(labels)} cells)")
            continue
        y = np.array([e[tag][l][0] for l in labels])
        err = y - x
        h = np.array([height(read(os.path.join(HERE, tag, "system", f"{l}.extxyz"))) for l in labels])
        s = dict(mae=np.abs(err).mean(), signed=err.mean(), mae_seen=np.abs(err[seen]).mean(),
                 mae_held=np.abs(err[~seen]).mean(), mae_held_main=np.abs(err[~seen & ~small]).mean(),
                 small=err[small].mean(), mean=y.mean(), h=h.mean(), dh=(h - h_dft).mean(),
                 notconv=sum(not e[tag][l][1] for l in labels),
                 broken=sum(e[tag][l][2] > 1 for l in labels))
        if tag == "m11nod4_d4":   # split post-hoc E_ads into D4 part and model (PBE) part
            mol = read(os.path.join(HERE, tag, "molecule", "isolated_adsorbate.extxyz"))
            dm = d4(mol)
            disp = np.array([d4(read(os.path.join(HERE, tag, "system", f"{l}.extxyz")))
                             - d4(read(os.path.join(HERE, tag, "substrate", f"{l}.extxyz"))) - dm
                             for l in labels])
            s.update(d4_part=disp.mean(), pbe_part=(y - disp).mean())
        stats[tag] = s
    pbe = np.array([e["DFT_PBE"][l][0] for l in labels])
    dft_d4_part, dft_pbe_part = (x - pbe).mean(), pbe.mean()

    # ---------------- report ----------------
    L = [f"# Adsorption energies with three ways of handling dispersion (model11 family)",
         "", f"Generated automatically {datetime.datetime.now():%Y-%m-%d %H:%M} by `make_report_m11.py`.",
         "", "![parity plot](eads_parity_s18_style_m11.png)", "",
         "Per-cell view (every cell on its own x position, so no point hides another):", "",
         "![per-cell plot](eads_per_cell_m11.png)", ""]
    if missing:
        L += ["> **Incomplete:** missing results for " + ", ".join(missing) +
              ". Numbers below cover only the finished series; rerun `make_report_m11.py` once they exist.", ""]

    L += ["## The three models", "",
          "All three share the same base (model_01 lineage, MACE, 5 Å cutoff, 2 layers) and the same "
          "fine-tuning structures: model11's DFT data for this system (relaxation snapshots of the "
          "bound systems and bare substrates of 12 training, 2 validation and 2 test cells, plus the "
          "isolated 126-atom molecule). They differ only in "
          "how dispersion enters:", ""]
    L += [f"- **{name}**: {desc}." for _, name, desc in SERIES]
    L += ["", "Every model relaxed the same 26 DFT starting structures with the DFT protocol: combined "
          "system relaxed with the substrate fixed, bare substrate single point at the same geometry, "
          "isolated molecule relaxed. E_ads = E_system − E_substrate − E_molecule. Reference: the DFT "
          "PBE+D4 adsorption energies of the same cells.", ""]

    if stats:
        L += ["## Results", "",
              "| Series | E_ads mean (eV) | MAE vs DFT (eV) | mean signed error (eV) | MAE, cells seen in "
              "fine-tuning (12) | MAE, held-out cells (14) | adsorption height vs DFT (Å) |",
              "|---|---|---|---|---|---|---|"]
        for tag, name, _ in SERIES:
            if tag in stats:
                s = stats[tag]
                L.append(f"| {name} | {s['mean']:+.2f} | **{s['mae']:.2f}** | {s['signed']:+.2f} | "
                         f"{s['mae_seen']:.2f} | {s['mae_held']:.2f} | {s['dh']:+.2f} |")
        L.append(f"| DFT (PBE+D4) | {x.mean():+.2f} | — | — | — | — | (height {h_dft.mean():.2f} Å) |")
        L += ["", "For comparison, the model_01-based trio (no adsorption data for this molecule in "
              "training) gave MAEs of 6.35 / 5.54 / 4.12 eV (no / learned / post-hoc).", ""]

        L += ["## What the dispersion treatment does", ""]
        ranked = sorted(stats, key=lambda t: stats[t]["mae"])
        names = {t: n for t, n, _ in SERIES}
        L.append("- **Ranking by error:** " + " < ".join(f"{names[t]} ({stats[t]['mae']:.2f} eV)"
                                                       for t in ranked) + ".")
        if "m11nod4" in stats and len(stats) == 3:
            better = all(stats[t]["mae"] < stats["m11nod4"]["mae"] for t in ("m11_d4trained", "m11nod4_d4"))
            L.append("- **Both dispersion-handling models are closer to DFT than the dispersion-free one: "
                     + ("yes.**" if better else "no.** Check the per-cell results before drawing conclusions."))
            s0 = stats["m11nod4"]
            L.append(f"- **Without dispersion** the model binds by {s0['mean']:+.2f} eV on average against "
                     f"DFT's {x.mean():+.2f} eV, an under-binding of {s0['signed']:+.2f} eV. In DFT, the D4 "
                     f"share of E_ads is {dft_d4_part:+.2f} eV and the PBE share {dft_pbe_part:+.2f} eV, so "
                     "the dispersion-free gap should be roughly the D4 share. The molecule sits "
                     f"{s0['dh']:+.2f} Å from the DFT height.")
        if "m11nod4_d4" in stats and "d4_part" in stats["m11nod4_d4"]:
            s = stats["m11nod4_d4"]
            L.append(f"- **Post-hoc D4** splits into a D4 part of {s['d4_part']:+.2f} eV (DFT: "
                     f"{dft_d4_part:+.2f}) and a model (PBE) part of {s['pbe_part']:+.2f} eV (DFT: "
                     f"{dft_pbe_part:+.2f}). The remaining error lies mostly in "
                     + ("the dispersion part." if abs(s['d4_part'] - dft_d4_part) > abs(s['pbe_part'] - dft_pbe_part)
                        else "the model's PBE part, not in the dispersion."))
        if "m11_d4trained" in stats and "m11nod4_d4" in stats:
            a, b = stats["m11_d4trained"], stats["m11nod4_d4"]
            diff = b["mae"] - a["mae"]
            verdict = ("about as accurate as" if abs(diff) < 0.03 else
                       ("more accurate than" if diff < 0 else "less accurate than"))
            L.append(f"- **Learned vs post-hoc:** post-hoc is {verdict} learned "
                     f"({b['mae']:.2f} vs {a['mae']:.2f} eV). The learned model only sees dispersion within "
                     "10 Å; the post-hoc model includes the full-range tail, which for these cells is a "
                     "near-uniform ~−0.1 eV (earlier tail measurement, eads_vs_R).")
        L += ["", "## Caveats", "",
              "- 12 of the 26 cells were in the fine-tuning data; 14 were not. The plot does not distinguish "
              "them; the table does. Held-out errors are the fairer measure.",
              f"- The two 234-atom cells ({', '.join(sorted(SMALL))}) are a cell type absent from the "
              "fine-tuning data; mean error on them: " + ", ".join(
                  f"{names[t]} {stats[t]['small']:+.2f} eV" for t in stats) + ". Held-out MAE without "
              "them: " + ", ".join(f"{names[t]} {stats[t]['mae_held_main']:.2f} eV" for t in stats) + ".",
              "- The DFT pure-PBE binding (~−2 eV at the PBE+D4 geometry) has not been independently "
              "checked (open question); it sets where the no-dispersion points should land.",
              "- The post-hoc series uses D4, the same dispersion as the DFT reference. With D3 it would be "
              "compared against a reference with a different dispersion model.", ""]
        L += ["## Run status", ""] + [
            f"- {names[t]}: {stats[t]['notconv']} relaxations not converged, {stats[t]['broken']} "
            "fragmented adsorbates." for t in stats]
    L += ["", "<details><summary>pipeline log</summary>", "", "```", *log, "```", "</details>", ""]

    path = os.path.join(OUT, "report_model11_dispersion.md")
    open(path, "w").write("\n".join(L))
    for f in ("eads_parity_s18_style_m11.png", "eads_per_cell_m11.png", "eads_all_variants.csv"):
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy(os.path.join(HERE, f), OUT)
    print(f"wrote {path}  (missing: {missing or 'none'})")


if __name__ == "__main__":
    main()
