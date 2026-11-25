#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/vggsound/fanchor_%j.out
#SBATCH -e logs_cluster/vggsound/fanchor_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

HIER_CONDS="False True"
LAMS="0.1 0.5 1.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0"
SEEDS="0 1 2 3 4"

for hier_cond in $HIER_CONDS; do
for lam in $LAMS; do
    ARGS="hier_cond=$hier_cond lam=$lam"
    for seed in $SEEDS; do
        echo "=== ${ARGS} seed=${seed} ==="
        python train.py vggsound fanchor $ARGS train_seed=$seed
    done
    python summarize.py vggsound fanchor $ARGS summary_seeds=[${SEEDS// /,}]
done
done
