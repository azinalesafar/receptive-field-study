#!/bin/bash -l
#SBATCH --job-name=sm_mace
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-114
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/small_molecules
#SBATCH --output=slurm_logs/mace_%a_%A.out
# 4 lines of tasks_mace.txt per array task, run side by side (8 threads each); same relaxation
# script and protocol as the candidate cells (../dispersion_inference/relax_with_dispersion.py).
module load gpaw/25.7.0
source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
export OMP_NUM_THREADS=8
RELAX=/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference/relax_with_dispersion.py
run_line() {
    local n=$1
    read -r VARIANT KIND NAME INPUT MODEL DISP CUT MODE < <(sed -n "${n}p" tasks_mace.txt)
    [ -z "$VARIANT" ] && return 0
    python3 -u $RELAX "$INPUT" --name "$NAME" --outdir "mace/$VARIANT/$KIND" --model "$MODEL" \
        --dispersion "$DISP" --mode "$MODE" > "slurm_logs/mace_line_${n}.log" 2>&1
    echo "line $n exit $?"
}
for k in 1 2 3 4; do run_line $((4 * SLURM_ARRAY_TASK_ID + k)) & done
wait
