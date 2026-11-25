#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --mem=192G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/vggsound/repercent_%j.out
#SBATCH -e logs_cluster/vggsound/repercent_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

ALPHA_LMDS="2:1 6:1 6:4 10:4"
SEEDS="0 1 2 3 4"

for alpha_lmd in $ALPHA_LMDS; do
alpha=${alpha_lmd%%:*}; lmd=${alpha_lmd##*:}
    ARGS="alpha=$alpha lmd=$lmd"
    for seed in $SEEDS; do
        echo "=== ${ARGS} seed=${seed} ==="
        python train.py vggsound repercent $ARGS train_seed=$seed
    done
    python summarize.py vggsound repercent $ARGS summary_seeds=[${SEEDS// /,}]
done
