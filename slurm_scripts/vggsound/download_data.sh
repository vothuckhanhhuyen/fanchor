#!/bin/bash
#SBATCH --partition=a100
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/vggsound/download_data_%j.out
#SBATCH -e logs_cluster/vggsound/download_data_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

huggingface-cli download Loie/VGGSound --repo-type dataset --local-dir data/vggsound
mkdir -p data/vggsound/video
for f in data/vggsound/vggsound_*.tar.gz; do
    tar xzf $f -C data/vggsound/video --strip-components=6
done

huggingface-cli download facebook/pe-av-base-16-frame --cache-dir data/vggsound/.cache/pe_av
