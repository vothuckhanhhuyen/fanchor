#!/bin/bash
#SBATCH --partition=a100
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/clevr4/download_data_%j.out
#SBATCH -e logs_cluster/clevr4/download_data_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p data/clevr_4_100k
wget -P data https://thor.robots.ox.ac.uk/clevr4/clevr_4_100k_v1.zip
unzip -q data/clevr_4_100k_v1.zip -d data/clevr_4_100k
