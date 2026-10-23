#!/bin/bash -l
#SBATCH --job-name=hs_dft
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --array=0-12
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/height_scan
#SBATCH --output=slurm_logs/dft_%a_%A.out
# DFT single points (GPAW PW500/PBE/k221 + D4, pbc T T F, 37 A box) for the height scan:
# 11 lifted systems, the substrate and the frozen molecule alone (dft_inputs.txt).
# Same calculator script as ../multilayer_test.
module load gpaw/25.7.0
export OMP_NUM_THREADS=1
IN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dft_inputs.txt)
echo "task $SLURM_ARRAY_TASK_ID: $IN on $(hostname)"
srun gpaw python ../multilayer_test/dft_calc.py sp "$IN" dft
