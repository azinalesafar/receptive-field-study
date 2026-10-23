# Height scan: explicit cutoff effect of learned vs post-hoc dispersion

Follows `../multilayer_test` (which could not isolate the cutoff: adding layers is out of
distribution for every model).

**Idea.** Lift the DFT-relaxed molecule of cell `m-8_n0_m-6_n-8` rigidly off single-layer graphene
and compute E_int = E_system − E_substrate − E_molecule(frozen, same cell) at each height.
model11 has r_max = 5 Å and 2 interactions; across the empty gap there are no intermediate atoms,
so once the shortest molecule–graphene distance exceeds 5 Å the learned model's E_int is exactly 0.
DFT (PBE+D4) and post-hoc D4 keep a ~1/d⁴ tail.

**Setup.** Lifts dh = 0–8 Å; box enlarged 24 → 37 Å (graphene at the same z, ≥ 10.9 Å vacuum above
the molecule). DFT: GPAW PW500/PBE/k221 + D4 via `../multilayer_test/dft_calc.py`.

Files: `build_geometries.py` → `geometries/`; `job_dft.slurm` → `dft/`; `run_mace.sh` → `mace/`
(login node, 4 threads: the D4 variant segfaults on compute nodes); `compute_eint.py` →
`eint_height_scan.csv`, `eint_height_scan.png`.
