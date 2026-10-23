#!/bin/bash
#SBATCH --job-name=d4_tail_slab
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Measures how much DFT-D4 dispersion lies beyond MACE's 10 Ang receptive field
# for the production slab/water system, to decide whether stripping D4 from the
# training labels and re-adding it at inference would recover anything real.
#
# Read-only: reads structures and trajectories, writes only into this directory.

module load gpaw/25.7.0

# dftd4's Fortran core overflows the OpenMP *thread* stack on these systems, and
# `ulimit -s` alone does not govern that. The script re-execs itself to set
# OMP_STACKSIZE/OMP_NUM_THREADS, but the stack rlimit has to be raised here.
ulimit -s unlimited

cd /scratch/johnt447/azin/analysis/d4_strip_test

echo "host       : $(hostname)"
echo "cpus       : ${SLURM_CPUS_PER_TASK}"
echo "start      : $(date)"
echo

srun python3 -u measure_d4_tail_slab.py \
    --frames 10 \
    --adsorption \
    --bulk-water-check \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --save --plots --force

STATUS=$?
echo
echo "end        : $(date)"
echo "exit       : ${STATUS}"
exit ${STATUS}
