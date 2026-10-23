#!/bin/bash
#SBATCH --job-name=d4_tail_vs_R
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# E_tail as a continuous function of cutoff radius, 10 Ang -> D4's 31.75 Ang
# two-body limit, for the production slab/water system. Answers "what would a
# larger receptive field actually recover?", which the coarse V9 ladder in
# measure_d4_tail_slab.py (capped at 16 Ang) cannot show.
#
# Read-only w.r.t. the research tree: reads the trajectory and the existing
# d4_tail_slab.npz, writes only into this directory.
#
# Sizing: an independent ball per (atom, radius) -- 50 atoms x 25 radii = 1250
# D4 calls. Measured cost is ~N^2.55 and ball occupancy runs 290 atoms at
# R = 10 to 4717 mean / 8721 worst at R = 31.75, giving ~23000 core-seconds in
# total; the biggest balls are handed out first so one 8721-atom ball cannot set
# the wall time on its own. Each worker holds two (N,N) pairwise matrices,
# 1.2 GB together at the worst ball, hence the large --mem rather than the 64G
# the sibling job used.
#
# The tempting one-call-per-atom shortcut (cumulative sum over a big ball's
# pairwise row) was implemented, measured and rejected -- it is wrong for the
# three-body term. See the module docstring before trying to speed this up.

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

srun python3 -u measure_d4_tail_vs_R.py \
    --per-region 10 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --save --plots

STATUS=$?
echo
echo "end        : $(date)"
echo "exit       : ${STATUS}"
exit ${STATUS}
