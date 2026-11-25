#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH --array=0-5
#SBATCH -o logs_cluster/clevr4/jointdisen_%A_%a.out
#SBATCH -e logs_cluster/clevr4/jointdisen_%A_%a.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

SETTINGS_ALL=(image:text image_no_color:text image:text_no_shape image_no_color:text_no_shape image:text_no_texture image_no_color:text_no_texture)
SETTINGS=${SETTINGS_ALL[$SLURM_ARRAY_TASK_ID]}

EMBED_DIMS="4 8 16 32 64"
AS="0.01 0.1 1 10 100"
LMDS="0 0.001 0.1"
SEEDS="0 1 2 3 4"

for setting in $SETTINGS; do
image=${setting%%:*}; text=${setting##*:}
for embed in $EMBED_DIMS; do
for a in $AS; do
for lmd in $LMDS; do
    ARGS="image=$image text=$text embed_dim=$embed private_dim=$embed a=$a lmd=$lmd"
    for seed in $SEEDS; do
        echo "=== ${ARGS} seed=${seed} ==="
        python train.py clevr4 jointdisen $ARGS train_seed=$seed
    done
    python summarize.py clevr4 jointdisen $ARGS summary_seeds=[${SEEDS// /,}]
done
done
done
done
