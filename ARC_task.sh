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

# Load the required setup module first
module load apps site/tinkercliffs/easybuild/setup

# Now you can load Anaconda3/2020.11
module load Anaconda3/2020.11

# Activate your Python environment
source activate pytorch

# Run your script
python yolov5_lightning.py

# in terminal, run `sbatch ARC_task.sh`
