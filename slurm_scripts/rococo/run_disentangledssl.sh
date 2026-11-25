#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/rococo/disentangledssl_%j.out
#SBATCH -e logs_cluster/rococo/disentangledssl_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

LMDS="0 0.001 0.1"
BETAS="0.01 0.1 1 10 100"
SEEDS="0 1 2 3 4"

for lmd in $LMDS; do
for beta in $BETAS; do
    ARGS="lmd=$lmd beta=$beta"
    for seed in $SEEDS; do
        echo "=== ${ARGS} seed=${seed} ==="
        python train.py rococo disentangledssl $ARGS train_seed=$seed
    done
    python summarize.py rococo disentangledssl $ARGS summary_seeds=[${SEEDS// /,}]
done
done
