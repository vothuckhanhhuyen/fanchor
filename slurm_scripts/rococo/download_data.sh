#!/bin/bash
#SBATCH --partition=a100
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o logs_cluster/rococo/download_data_%j.out
#SBATCH -e logs_cluster/rococo/download_data_%j.err

source ~/.bashrc
conda activate fanchor

cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p data/rococo/MSCOCO/images
ln -sfn ../../third_party/rococo/annotations data/rococo/annotations

I=data/rococo/MSCOCO/images
wget -P $I http://images.cocodataset.org/zips/train2014.zip http://images.cocodataset.org/zips/val2014.zip
unzip -q $I/train2014.zip -d $I
unzip -q $I/val2014.zip -d $I

for m in mix patch; do
for l in 0.9 0.8; do
for s in 1 10 100; do
    python third_party/rococo/mix_images.py --method $m --lam $l --seed $s --data_path $I/ --img_list third_party/rococo/annotations/coco_karpathy_test.json
    mv $I/${m}_${l} $I/${m}_${l}_seed${s}
done
done
done
