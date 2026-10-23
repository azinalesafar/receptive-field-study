#!/bin/bash -l
#SBATCH --job-name=sm_gate
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/small_molecules
#SBATCH --output=slurm_logs/gate_%j.out
# Fails (exit 1) unless the recomputed model_01 end point matches its stored energy within 5 meV,
# so the molecule relaxations (afterok on this job) never start on mismatched DFT settings.
module load gpaw/25.7.0
python3 - <<'PY'
import sys
from ase.io import read
try:
    a = read('systems/r00_C76H6N2.extxyz'); b = read('dft/check/r00_C76H6N2.extxyz')
    read('dft/graphene/graphene.extxyz')
except Exception as e:
    print('GATE FAIL: missing output', e); sys.exit(1)
d = 1000 * (b.get_potential_energy() - a.get_potential_energy())
print(f'stored {a.get_potential_energy():.6f}  recomputed {b.get_potential_energy():.6f}  diff {d:+.3f} meV')
sys.exit(0 if abs(d) < 5 else 1)
PY
