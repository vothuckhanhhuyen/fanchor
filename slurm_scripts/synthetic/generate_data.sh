#!/bin/bash
#SBATCH --partition=a100
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH -o logs_cluster/synthetic/generate_data_%j.out
#SBATCH -e logs_cluster/synthetic/generate_data_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

python -m datasets.synthetic_data
python -m datasets.synthetic_data --n_modalities 3
