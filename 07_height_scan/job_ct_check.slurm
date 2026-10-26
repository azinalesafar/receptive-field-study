#!/bin/bash -l
#SBATCH --job-name=hs_ct
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/height_scan
#SBATCH --output=slurm_logs/ct_check_%a_%A.out
# Charge-transfer check (dft_ct_check.py): lift 9 A system + substrate + molecule, for two settings.
module load gpaw/25.7.0
export OMP_NUM_THREADS=1
SETTINGS=(w001 w001 w001 w001_spin w001_spin w001_spin)
INPUTS=(scan_dh9.0 substrate molecule scan_dh9.0 substrate molecule)
S=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
IN=geometries/${INPUTS[$SLURM_ARRAY_TASK_ID]}.extxyz
echo "task $SLURM_ARRAY_TASK_ID: $S $IN on $(hostname)"
srun gpaw python dft_ct_check.py "$S" "$IN"
