#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=unittest
#SBATCH --output=out.log
#SBATCH --error=out.err

module load Anaconda3/2020.11
source activate pytorch
python yolov5_lightning.py

# in terminal, run `sbatch ARC_task.sh`
