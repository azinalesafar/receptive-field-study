#!/bin/bash
# MACE single points for the height scan, three model11-family variants (as ../multilayer_test).
# Run on the login node with 4 threads: the D4 variant segfaults (exit 139) on compute nodes
# with 32 threads (jobs 2429360, 2462756).
module load gpaw/25.7.0
source /scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp/bin/activate
export OMP_NUM_THREADS=4
cd /scratch/johnt447/azin/analysis/ads_eng_calc/height_scan
RELAX=/scratch/johnt447/azin/analysis/ads_eng_calc/dispersion_inference/relax_with_dispersion.py
M11=/scratch/johnt447/azin/SPICE_MACE/model11_adsorption_dft_activelearning/model11_stagetwo.model
M11N=/scratch/johnt447/azin/SPICE_MACE/model11_no_d4/model11_no_d4_stagetwo.model
for spec in "m11_d4trained $M11 none" "m11nod4 $M11N none" "m11nod4_d4 $M11N d4"; do
    set -- $spec
    for f in $(cat dft_inputs.txt); do
        b=$(basename "$f" .extxyz)
        [ -f "mace/$1/$b.extxyz" ] && continue
        python3 -u $RELAX "$f" --name "$b" --outdir "mace/$1" --model "$2" --dispersion "$3" --mode sp \
            2>&1 | grep -E "^E =|rror"
    done
done
