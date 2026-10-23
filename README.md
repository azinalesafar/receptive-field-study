# Receptive field and dispersion in a MACE potential for porphyrin on graphene

This repository collects the tests showing that, for a porphyrin (C56H62N8) adsorbed on graphene,
**increasing the receptive field of the MACE potential beyond 10 Å does not improve the results**,
except when the molecule is lifted away from the surface.

## Setup

- **Model:** MACE, r_max = 5 Å, 2 message-passing layers (nominal receptive field 10 Å; across a
  vacuum gap the model only sees 5 Å, because there are no atoms in between to pass messages).
- **Reference:** DFT, GPAW PW 500 eV / PBE / k 2×2×1 + D4 (D4 two-body range ≈ 32 Å).
- **Three model variants** (tests 6–10):
  - *no vdW*: model trained on PBE labels (D4 stripped).
  - *learned vdW (10 Å)*: model trained on PBE+D4 labels; dispersion is learned inside the 10 Å receptive field.
  - *post-hoc D4*: the *no vdW* model with D4 added at inference (≈ 32 Å range).

## Tests

Each folder contains the scripts, the Slurm job file, and the results (summary, CSV, plots).

### Part 1 — How much dispersion lies beyond the 10 Å receptive field?

| # | Folder | Question | Result | Conclusion |
|---|---|---|---|---|
| 1 | `01_d4_tail_full_system` | How large are the D4 energy and forces beyond 10 Å, compared with the model's own error? | Force error from the tail 0.33 meV/Å, 76× below the model error (24.8 meV/Å); 81% of the tail is predictable from the model's 10 Å features | The 10 Å receptive field loses nothing meaningful |
| 2 | `02_tail_vs_cutoff` | How fast does the missing D4 energy decay as the cutoff grows from 10 to 32 Å? | 0.58 meV/atom missing at 10 Å; going to 20 Å recovers 90% of it | Small, and mostly a constant that training absorbs |
| 3 | `03_dry_slab_d3_vs_cutoff` | Same question without water, using D3 up to 50 Å | 0.60 meV/atom missing at 10 Å, 0.12 at 15 Å | Same small numbers without water |

### Part 2 — Does the missing tail change the adsorption energies?

| # | Folder | Question | Result | Conclusion |
|---|---|---|---|---|
| 4 | `04_eads_ranking_with_without_tail` | Does ignoring dispersion beyond 10 Å change which of the 26 cells binds strongest? | Tail −97.7 ± 0.4 meV, almost identical for every cell; 0 of 325 pairs change order | Ranking unchanged |
| 5 | `05_eads_tail_vs_cutoff` | Does the ranking change at any cutoff between 10 and 32 Å? | Lattice-cluster order never changes; only one nearly tied pair (DFT gap 0.36 meV) swaps | A larger cutoff does not change the result |

### Part 3 — Learned (10 Å) vs post-hoc D4 (≈ 32 Å), against DFT

| # | Folder | Result | Conclusion |
|---|---|---|---|
| 6 | `06_porphyrin_eads_learned_vs_posthoc` | Porphyrin, 26 cells: E_ads MAE 0.14 eV (learned) vs 0.14 eV (post-hoc); per-cell difference −0.9 ± 15 meV | No improvement from the longer range |
| 7 | `07_small_molecules` | Small molecules on graphene, 57 runs: MAE 0.39 eV (learned) vs 0.46 eV (post-hoc) | No improvement |
| 8 | `08_longrange_screen` | D3 screen of geometries where long range should matter most (multilayers, stacked and upright molecules): only 1.5–7% of E_ads comes from beyond 10 Å | No realistic geometry has a large long-range share |
| 9 | `09_multilayer_graphene` | Porphyrin on 2 and 3 graphene layers: extra binding from layer 2, error vs DFT −346 meV (learned) vs −350 meV (post-hoc) | No improvement; both models fail on the unseen multilayer substrate, so this test is weak evidence either way |

### The exception

| # | Folder | Result | Conclusion |
|---|---|---|---|
| 10 | `10_height_scan` | Molecule lifted rigidly off the graphene: beyond a 5 Å gap the learned model gives exactly zero interaction, while post-hoc D4 keeps an attractive tail (−229 meV at 5.3 Å, −21 meV at 10.3 Å). DFT reference pending. | The receptive field matters only away from the adsorbed geometry |

## Caveats

- Test 1 measures **dispersion only**; long-range electrostatics were not tested.
- An earlier estimate of the bulk-water tail (0.03%, file `tail3.txt`, not included) is wrong;
  the correct value is 1.85% of E_D4 (see test 1).

## Reproducing

The scripts were run on the CSC Roihu cluster and contain absolute paths to the original project
directories (`/scratch/johnt447/azin/...`), models and DFT data, which are not part of this
repository. Python environment: GPAW 25.7.0 module + mace-torch 0.3.16, torch-dftd, dftd4 3.7
(see `06_porphyrin_eads_learned_vs_posthoc/setup_env.sh`).

## Related work

- Solé et al., *Machine Learning Multiscale Interactions*, arXiv:2605.25710 (2026): a 5 Å
  SO3krates model underestimates the naphthyridine–graphene binding at intermediate separations;
  fixed there with a multiscale architecture rather than a post-hoc dispersion term.
