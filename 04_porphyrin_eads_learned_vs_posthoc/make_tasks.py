"""Write tasks.txt: one line per (variant, structure) job-array task.

Columns: variant  kind  name  input  model  dispersion  disp_cutoff(- = full)  mode
All inputs are the DFT inputs (same start geometries the DFT E_ads came from).
"""
import csv

ROOT = "/scratch/johnt447/azin/analysis/ads_eng_calc"
M01 = f"{ROOT}/MACE_model01/MP_to_organic_graphene_run-345_stagetwo.model"
M01_NOD4 = ("/scratch/johnt447/azin/SPICE_MACE/model_01/model_01_no_d4/models/"
            "MP_to_organic_graphene_no_d4_stagetwo.model")

M11 = "/scratch/johnt447/azin/SPICE_MACE/model11_adsorption_dft_activelearning/model11_stagetwo.model"
M11_NOD4 = "/scratch/johnt447/azin/SPICE_MACE/model11_no_d4/model11_no_d4_stagetwo.model"

VARIANTS = [  # tag, model, dispersion, cutoff
    ("m01_d4trained", M01, "none", "-"),
    ("m01nod4", M01_NOD4, "none", "-"),
    ("m01nod4_d3", M01_NOD4, "d3", "-"),
    ("m01nod4_d3_15", M01_NOD4, "d3", "15.0"),
    ("m01nod4_d4", M01_NOD4, "d4", "-"),
]

# model11 family (2026-10-20): separate task files so the no_d4 ones can wait for training
M11_GROUPS = {
    "tasks_m11_trained.txt": [("m11_d4trained", M11, "none", "-")],
    "tasks_m11_nod4.txt": [("m11nod4", M11_NOD4, "none", "-"), ("m11nod4_d4", M11_NOD4, "d4", "-")],
}

labels = [r["label"] for r in csv.DictReader(open(f"{ROOT}/DFT/adsorption_energies.csv"))]
rows = []
for tag, model, disp, cut in VARIANTS:
    rows.append((tag, "molecule", "isolated_adsorbate",
                 f"{ROOT}/DFT/isolated_adsorbate/isolated_adsorbate.extxyz", model, disp, cut, "relax"))
    for l in labels:
        rows.append((tag, "system", l, f"{ROOT}/DFT/unitcell_candidates/{l}.traj", model, disp, cut, "relax"))
        rows.append((tag, "substrate", l, f"{ROOT}/DFT/substrate/{l}.traj", model, disp, cut, "sp"))
# reproduction check: old model_01 inputs, must match MACE_model01/adsorption_energies.csv
for l in ["m-8_n0_m-6_n-8", "m-1_n-8_m-8_n-5"]:
    rows.append(("repro_m01", "system", l, f"{ROOT}/MACE_model01/unitcell_candidates/{l}.traj",
                 M01, "none", "-", "relax"))
import sys
if "--m11" not in sys.argv:
    with open("tasks.txt", "w") as f:
        for r in rows:
            f.write(" ".join(r) + "\n")
    print(f"{len(rows)} tasks")
else:
    for fn, group in M11_GROUPS.items():
        with open(fn, "w") as f:
            n = 0
            for tag, model, disp, cut in group:
                f.write(" ".join((tag, "molecule", "isolated_adsorbate",
                        f"{ROOT}/DFT/isolated_adsorbate/isolated_adsorbate.extxyz", model, disp, cut, "relax")) + "\n")
                n += 1
                for l in labels:
                    f.write(" ".join((tag, "system", l, f"{ROOT}/DFT/unitcell_candidates/{l}.traj", model, disp, cut, "relax")) + "\n")
                    f.write(" ".join((tag, "substrate", l, f"{ROOT}/DFT/substrate/{l}.traj", model, disp, cut, "sp")) + "\n")
                    n += 2
        print(f"{fn}: {n} tasks")
