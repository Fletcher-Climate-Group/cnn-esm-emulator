#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:0
#SBATCH --job-name=nt_100
#SBATCH --output=/scratch/c/cgf/mcnallyw/experiments/%x.out
#SBATCH -p compute_full_node

module load anaconda3
source activate py3.7-tf2.4-torch1.7
source deactivate py3.7-tf2.4-torch1.7
source activate py3.7-tf2.4-torch1.7
python tf/train_multi_res.py \
  --data-dir $SCRATCH/data \
  --exp-dir $SCRATCH/experiments \
  --exp-name nt_100 \
  --n-gpus 4 \
  --n-trials 40 \
  --n-test 20 \
  --n-train 100 \
  --n-hr 0 1 2 3 4 5 10 15 20 30 40 50 60 70 80 \
  --save-models