#!/bin/bash -l
#SBATCH --job-name=sm_check
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/small_molecules
#SBATCH --output=slurm_logs/check_graphene_%j.out
# 1) settings check: recompute one stored model_01 end point; must reproduce its stored energy
# 2) bare C72 graphene single point (same sheet in every frame, same cell, k 2 2 1)
module load gpaw/25.7.0
export OMP_NUM_THREADS=1
srun gpaw python dft_calc.py sp systems/r00_C76H6N2.extxyz dft/check
srun gpaw python dft_calc.py sp graphene.extxyz dft/graphene
python3 -c "
from ase.io import read
a=read('systems/r00_C76H6N2.extxyz'); b=read('dft/check/r00_C76H6N2.extxyz')
d=b.get_potential_energy()-a.get_potential_energy()
print(f'SETTINGS CHECK: stored {a.get_potential_energy():.6f}  recomputed {b.get_potential_energy():.6f}  diff {1000*d:+.3f} meV')"
