#!/bin/bash
#SBATCH --job-name=d4_tail_vs_R_all
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Whole-system E_tail(R), 10 Ang -> 31.75 Ang, using EVERY atom (2087) of frame
# 200 rather than the 10-per-region sample of measure_d4_tail_vs_R.py.
#
# Sizing: 2087 atoms x 25 radii = 52175 independent D4 balls. The 50-atom run
# (job 1452416) did 1250 balls in 1174 s on 50 workers, i.e. ~59000 core-s,
# so this is ~2.4M core-s = ~5.5 h on 128 workers if scaling holds.
# Memory: that run peaked at 152 GB for 50 workers (~3 GB each, two (N,N)
# pairwise matrices at the largest balls), so 128 workers need ~390 GB.
# In practice 128 workers OOM'd at 480 GB within ~1 min (jobs 1558101/220/221):
# balls run biggest-first, so all workers hold ~12.5k-atom balls at once.
# Now 64 workers (WORKERS overrides) on the full 128-core node; ~11 h expected.
#
# Checkpointed every 10 min to checkpoint.npz. If the job dies or times out,
# resubmit this same file: it resumes from the checkpoint.

module load gpaw/25.7.0

# dftd4's Fortran core overflows the OpenMP thread stack otherwise (the script
# also re-execs itself to set OMP_STACKSIZE / OMP_NUM_THREADS).
ulimit -s unlimited

cd /scratch/johnt447/azin/analysis/d4_strip_test/all_atoms_vs_R

echo "host       : $(hostname)"
echo "cpus       : ${SLURM_CPUS_PER_TASK}"
echo "start      : $(date)"
echo

# FRAME_SLOT picks one of the 10 frames stored in d4_tail_slab.npz
# (0 = frame 200, 5 = 933, 9 = 1520). Default 0. Submit another frame with
#   sbatch --export=ALL,FRAME_SLOT=5 -J d4_tail_vs_R_all_f933 job_d4_tail_vs_R_all_atoms.slurm
echo "frame slot : ${FRAME_SLOT:-0}"
echo "workers    : ${WORKERS:-64}"

srun python3 -u measure_d4_tail_vs_R_all_atoms.py \
    --workers "${WORKERS:-64}" \
    --frame-slot "${FRAME_SLOT:-0}" \
    --save --force

STATUS=$?
echo
echo "end        : $(date)"
echo "exit       : ${STATUS}"
exit ${STATUS}
