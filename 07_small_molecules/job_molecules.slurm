#!/bin/bash -l
#SBATCH --job-name=sm_mol
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/small_molecules
#SBATCH --output=slurm_logs/mol_%a_%A.out
# One isolated-molecule DFT relaxation per distinct molecule (molecules/*.xyz, sorted), set up
# like DFT/isolated_adsorbate (vacuum 8 A, pbc, Gamma, PreconLBFGS fmax 0.01). Skips finished ones.
module load gpaw/25.7.0
export OMP_NUM_THREADS=1
mapfile -t MOLS < <(ls molecules/*.xyz | sort)
IN=${MOLS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: $IN on $(hostname)"
srun gpaw python dft_calc.py relax "$IN" dft/molecules
