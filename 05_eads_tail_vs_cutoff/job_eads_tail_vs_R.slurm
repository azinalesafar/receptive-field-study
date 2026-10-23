#!/bin/bash
#SBATCH --job-name=eads_tail_vs_R
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Missing E_ads vs cutoff radius (10 -> 31.75 Ang) for the 26 adsorbate-substrate
# structures, and whether any cutoff changes their DFT ranking.
#
# Sizing: 26 systems + 26 substrates + adsorbate x2 = 9872 atoms x 25 radii
# = 246800 balls, largest ~2400 atoms. The frame-1520 dry slab ran at ~1.9
# core-s per ball, so ~470k core-s = ~1 h on 128 workers.
#
# Checkpointed every 5 min; resubmit this file to resume.

module load gpaw/25.7.0
ulimit -s unlimited

cd /scratch/johnt447/azin/analysis/d4_strip_test/eads_vs_R

echo "host       : $(hostname)"
echo "cpus       : ${SLURM_CPUS_PER_TASK}"
echo "start      : $(date)"
echo

srun python3 -u measure_eads_tail_vs_R.py --workers "${WORKERS:-128}" --save --force

STATUS=$?
echo
echo "end        : $(date)"
echo "exit       : ${STATUS}"
exit ${STATUS}
