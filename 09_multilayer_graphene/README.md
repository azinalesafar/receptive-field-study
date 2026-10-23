# Multilayer test: does post-hoc (full-range) D4 beat learned (10 Å) dispersion?

Separate from the main parity study (`../dispersion_inference`, `../small_molecules`).

**Idea.** The learned-dispersion model (model11) only sees 10 Å and was trained on single-layer
graphene, so it has absorbed that case's average beyond-10 Å dispersion (≈ −0.09 eV per cell) but
cannot know about deeper graphene layers. A D3 screen (`../longrange_candidates`) gives an extra
beyond-10 Å contribution of ≈ −0.07 eV (bilayer) and ≈ −0.13 eV (trilayer) for the 126-atom
molecule. The post-hoc model (model11_no_d4 + D4, range ≈ 32 Å) includes it automatically.

**Protocol (single points, no relaxation).** For 4 candidate cells, the DFT-relaxed single-layer
geometry gets 1 or 2 extra AB-stacked graphene layers added below (3.35 Å spacing), in the same
24 Å box (non-periodic along z, as in the original DFT). DFT (GPAW PW500/PBE/k221 + D4) and the
three models are evaluated at exactly these geometries:
  E_ads = E_system(SP) − E_substrate(SP) − E_molecule(relaxed isolated, existing per method)
The 1-layer DFT values are the existing ones (same geometry, box and settings).

Files: `build_geometries.py` → `geometries/`; `job_dft.slurm` → `dft/`; `job_mace.slurm` → `mace/`;
`compute_eads.py` (after the jobs) → results.
