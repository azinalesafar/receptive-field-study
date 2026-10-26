# Height scan: explicit cutoff effect of learned vs post-hoc dispersion

Follows `../multilayer_test` (which could not isolate the cutoff: adding layers is out of
distribution for every model).

**Idea.** Lift the DFT-relaxed molecule of cell `m-8_n0_m-6_n-8` rigidly off single-layer graphene
and compute E_int = E_system − E_substrate − E_molecule(frozen, same cell) at each height.
model11 has r_max = 5 Å and 2 interactions; across the empty gap there are no intermediate atoms,
so once the shortest molecule–graphene distance exceeds 5 Å the learned model's E_int is exactly 0.
Post-hoc D4 keeps a ~1/d⁴ tail.

**Setup.** 21 lifts, dh = −0.3 to 9 Å (`SHIFTS` + `EXTRA_SHIFTS` in `build_geometries.py`); the
negative lifts and 0.1–0.3 Å resolve the well. Box enlarged 24 → 37 Å (graphene at the same z,
≥ 9.9 Å vacuum above the molecule). DFT: GPAW PW500/PBE/k221 + D4 via `../multilayer_test/dft_calc.py`
(jobs 2488833 and 2497123, all converged).

## Result

- Beyond 5 Å the learned model gives exactly 0; post-hoc D4 equals the D4 part of DFT exactly.
- The full DFT curve has an extra slowly decaying tail (PBE part −0.54 eV at 11 Å). It is charge
  transfer, not dispersion: the adsorbate (C56H62N8, tetrakis(4-trimethylammoniophenyl)porphyrin)
  is a 4+ cation modelled as neutral by choice, so its 4 extra electrons sit in high-lying levels
  (top levels −1.5 eV vs graphene Fermi level −3.3 eV) and ~0.3 e moves to graphene at every height
  (z-dipole grows linearly with dh). Not a smearing/spin artefact: `dft_ct_check.py`
  (job 2549195; FD width 0.01 eV, spin-polarized) gives the same PBE tail (−0.50 / −0.49 eV at dh 9).
- Reference choice matters: with the frozen-molecule-in-cell reference the models are 0.6 eV deeper
  than DFT at the minimum; with the relaxed isolated molecule (the E_ads convention of
  `../dispersion_inference/eads_per_cell_m11.png`) the well agrees to 0.1 eV.

## Files

- `build_geometries.py` → `geometries/`; `dft_inputs.txt` (13) and `dft_inputs_extra.txt` (10)
- `job_dft.slurm` → `dft/` (extra lifts: `sbatch --array=0-9 --export=ALL,LIST=dft_inputs_extra.txt job_dft.slurm`)
- `run_mace.sh` → `mace/` (login node, 4 threads: the D4 variant segfaults on compute nodes)
- `compute_eint.py` → `eint_height_scan.csv`, `eint_height_scan.png`
- `plot_fig4_style.py` → Fig. 4-style plot (Solé et al., arXiv:2605.25710), with DFT's D4 part dashed:
  - default: interaction energy → `eint_fig4_style.png`
  - `--isolated`: adsorption energy, relaxed isolated molecule as reference →
    `eads_profile_fig4_style.png`. The isolated molecule's D4 energy is cached in
    `dft/isolated_molecule_d4.json`; if that file is deleted, run `module load gpaw/25.7.0` first
    (dftd4 is needed to recompute it).
- `dft_ct_check.py`, `job_ct_check.slurm` → `dft_ct_check/` (charge-transfer check)
