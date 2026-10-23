#!/bin/bash
#SBATCH --job-name=d3_vs_R
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Missing PBE-D3(BJ) energy vs pair cutoff R, dry frame 1520, two-body and two-body + ATM.
# Pair cutoff and CN/ATM cutoff are varied separately (see the script docstring).
#   1. 2x2 (1016 atoms), R = 10-17 Ang: ~2 min.
#   2. 6x6 (9144 atoms), R = 10-50 Ang (half-cell ~51 Ang), reference 50.27 Ang, plus an
#      80 Ang check that the reference is converged. 70 dftd3 calls in parallel;
#      2-body ~1-4 min each, ATM up to ~15 min each (single-threaded, ~0.4 GB each).
# The 2x2 runs first so the 6x6 plot can overlay it. Old coupled-cutoff 2x2 results:
# 2x2_no_water_v1_cn_coupled_to_R/.

module load gpaw/25.7.0

# No srun: under srun the gpaw python initialises MPI in every pool worker and
# the pool never shuts down (job 1865253 hung after all 32 dftd3 calls finished).

cd /scratch/johnt447/azin/analysis/d4_strip_test/d3_vs_R

echo "host  : $(hostname)"
echo "start : $(date)"
echo

python3 -u measure_d3_vs_R.py --workers "${SLURM_CPUS_PER_TASK}" --tag 2x2_no_water
S1=$?

python3 -u measure_d3_vs_R.py --workers "${SLURM_CPUS_PER_TASK}" --tag 6x6_no_water \
    --structure ../all_atoms_vs_R/frame_1520/frame_1520_6x6.traj \
    --radii 10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 \
            18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 \
    --ref-check 80
S2=$?

echo
echo "end   : $(date)"
echo "exit  : 2x2=${S1} 6x6=${S2}"
