#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:0
#SBATCH --job-name=default
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

module load anaconda3
conda activate tf2.2
python tf/train_multi_res.py \
  --data-dir $SCRATCH/data \
  --exp-dir $SCRATCH/experiments \
  --exp-name default \
  --n-gpus 4 \
  --n-trials 40 \
  --n-test 20 \
  --n-hr 0 1 2 3 4 5 10 15 20 30 40 50 60 70 80