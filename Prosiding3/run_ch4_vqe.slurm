#!/bin/bash
#SBATCH --job-name=CH4_VQE
#SBATCH --partition=medium-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

#SBATCH --output=logs/ch4_vqe_%j.out
#SBATCH --error=logs/ch4_vqe_%j.err

echo "===================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start Time  : $(date)"
echo "===================================="

# Load environment
source ~/.bashrc
conda activate env-ml

# Pindah ke folder tempat sbatch dijalankan
cd $SLURM_SUBMIT_DIR

# Buat folder output jika belum ada
mkdir -p logs
mkdir -p executed_notebooks

# Jalankan notebook
jupyter nbconvert \
    --to notebook \
    --execute CH4_VQE_kupccsd.ipynb \
    --output executed_notebooks/CH4_VQE_${SLURM_JOB_ID}.ipynb

echo "===================================="
echo "Finish Time : $(date)"
echo "===================================="