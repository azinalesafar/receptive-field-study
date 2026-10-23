#!/bin/bash -l
#SBATCH --job-name=ml_dft
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --array=0-16
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/multilayer_test
#SBATCH --output=slurm_logs/dft_%a_%A.out
# DFT single points (GPAW PW500/PBE/k221 + D4, pbc as in file = T T F, same as the original
# candidate-cell DFT) for the bilayer/trilayer systems and substrates in dft_inputs.txt; the last
# line is a 1-layer system whose energy must reproduce the existing DFT value (settings check).
module load gpaw/25.7.0
export OMP_NUM_THREADS=1
IN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dft_inputs.txt)
echo "task $SLURM_ARRAY_TASK_ID: $IN on $(hostname)"
srun gpaw python dft_calc.py sp "$IN" dft
