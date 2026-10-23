#!/bin/bash -l
# One-time setup: x86 CPU env with MACE + dftd4 (from gpaw module) + torch-dftd (D3).
# dftd4 exists only in the x86 gpaw module, so everything runs on CPU nodes.
set -euo pipefail
module load gpaw/25.7.0
VENV=/scratch/johnt447/azin/SPICE_MACE/mace_env_cpu_disp
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mace-torch==0.3.16 torch-dftd
python3 -c "import mace, dftd4.ase, torch_dftd, torch; print('ok', torch.__version__)"
