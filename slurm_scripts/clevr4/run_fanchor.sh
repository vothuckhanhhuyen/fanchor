#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH --array=0-5
#SBATCH -o logs_cluster/clevr4/fanchor_%A_%a.out
#SBATCH -e logs_cluster/clevr4/fanchor_%A_%a.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

PAIRS_ALL=(image:text image_no_color:text image:text_no_shape image_no_color:text_no_shape image:text_no_texture image_no_color:text_no_texture)
pair=${PAIRS_ALL[$SLURM_ARRAY_TASK_ID]}
image=${pair%%:*}; text=${pair#*:}

LATENT_DIMS="4 8 16 32 64"
HIER_CONDS="False True"
LAMS="0.1 0.5 1.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0"
SEEDS="0 1 2 3 4"

for hier_cond in $HIER_CONDS; do
for latent_dim in $LATENT_DIMS; do
for lam in $LAMS; do
    ARGS="image=$image text=$text shared_dim=$latent_dim private_dim=$latent_dim hier_cond=$hier_cond lam=$lam"
    for seed in $SEEDS; do
        echo "=== ${ARGS} seed=${seed} ==="
        python train.py clevr4 fanchor $ARGS train_seed=$seed
    done
    python summarize.py clevr4 fanchor $ARGS summary_seeds=[${SEEDS// /,}]
done
done
done
