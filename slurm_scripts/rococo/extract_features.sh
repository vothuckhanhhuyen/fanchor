#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/rococo/extract_features_%j.out
#SBATCH -e logs_cluster/rococo/extract_features_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"
export TMPDIR=$PWD/.cache/tmp
mkdir -p $TMPDIR

python -m datasets.rococo_features
python -m datasets.rococo_features --tokens
python -m datasets.rococo_features --pre_projection
