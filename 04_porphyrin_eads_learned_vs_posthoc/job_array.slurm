#!/bin/bash -l
#SBATCH --job-name=eads_disp
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-133
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference
#SBATCH --output=slurm_logs/%a_%A.out
#SBATCH --error=slurm_logs/%a_%A.err

# Each array task runs TWO lines of tasks.txt (make_tasks.py) side by side, 16 threads
# each (small partition caps submissions at 200 jobs). x86 CPU: dftd4 exists only here.
module load gpaw/25.7.0
source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
export OMP_NUM_THREADS=16
TASKS=${TASKS:-tasks.txt}   # e.g. sbatch --export=ALL,TASKS=tasks_m11_trained.txt

run_line() {
    local n=$1
    read -r VARIANT KIND NAME INPUT MODEL DISP CUT MODE < <(sed -n "${n}p" "$TASKS")
    [ -z "$VARIANT" ] && return 0
    local CUTARG=()
    [ "$CUT" != "-" ] && CUTARG=(--disp-cutoff "$CUT")
    echo "line $n: $VARIANT $KIND $NAME $DISP $CUT $MODE"
    python3 -u relax_with_dispersion.py "$INPUT" --name "$NAME" --outdir "$VARIANT/$KIND" \
        --model "$MODEL" --dispersion "$DISP" --mode "$MODE" "${CUTARG[@]}" \
        > "slurm_logs/${TASKS%.txt}_line_${n}.log" 2>&1
    echo "line $n exit $?"
}

run_line $((2 * SLURM_ARRAY_TASK_ID + 1)) &
run_line $((2 * SLURM_ARRAY_TASK_ID + 2)) &
wait
