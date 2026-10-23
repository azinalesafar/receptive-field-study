#!/bin/bash -l
#SBATCH --job-name=eads_report
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference
#SBATCH --output=slurm_logs/report_m11_%j.out
#SBATCH --error=slurm_logs/report_m11_%j.err

# Runs after the model11_no_d4 relaxations (submit with --dependency=afterany:<relax array>):
# recomputes all E_ads, makes eads_parity_s18_style_m11.png, writes report_m11/report_model11_dispersion.md.
module load gpaw/25.7.0
source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u make_report_m11.py
