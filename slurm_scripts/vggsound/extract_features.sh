#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH -t 7-00:00:00
#SBATCH --array=0-7
#SBATCH -o logs_cluster/vggsound/extract_features_%A_%a.out
#SBATCH -e logs_cluster/vggsound/extract_features_%A_%a.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"
export TMPDIR=$PWD/.cache/tmp
mkdir -p $TMPDIR

python -m datasets.vggsound_features --part $SLURM_ARRAY_TASK_ID --num_parts $SLURM_ARRAY_TASK_COUNT
python -m datasets.vggsound_features --part $SLURM_ARRAY_TASK_ID --num_parts $SLURM_ARRAY_TASK_COUNT --tokens
