#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/rococo/factorcl_%j.out
#SBATCH -e logs_cluster/rococo/factorcl_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

SEEDS="0 1 2 3 4"

for seed in $SEEDS; do
    echo "=== seed=${seed} ==="
    python train.py rococo factorcl train_seed=$seed
done
python summarize.py rococo factorcl summary_seeds=[${SEEDS// /,}]
