#!/bin/bash -l
#SBATCH --job-name=ml_mace
#SBATCH --account=johnt447
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --chdir=/scratch/johnt447/azin/analysis/ads_eng_calc/multilayer_test
#SBATCH --output=slurm_logs/mace_%j.out
# MACE single points at the multilayer geometries for the three model11-family variants
# (same models/settings as ../dispersion_inference). Isolated molecules are reused from there.
module load gpaw/25.7.0
source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
export OMP_NUM_THREADS=32
RELAX=/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference/relax_with_dispersion.py
M11=/scratch/johnt447/azin/SPICE_MACE/model11_adsorption_dft_activelearning/model11_stagetwo.model
M11N=/scratch/johnt447/azin/SPICE_MACE/model11_no_d4/model11_no_d4_stagetwo.model
for spec in "m11_d4trained $M11 none" "m11nod4 $M11N none" "m11nod4_d4 $M11N d4"; do
    set -- $spec
    for f in geometries/*.extxyz; do
        b=$(basename "$f" .extxyz)
        python3 -u $RELAX "$f" --name "$b" --outdir "mace/$1" --model "$2" --dispersion "$3" --mode sp \
            2>&1 | grep -E "^E =|Error"
    done
done
