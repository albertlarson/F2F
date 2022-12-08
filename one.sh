#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=1
#SBATCH --mem=16000
#SBATCH -o job-%j.out  # %j = job ID

module purge

eval "$(conda shell.bash hook)"

conda info --envs

conda activate /work/albertl_uri_edu/.conda/envs/sm


python -u one.py  > one.out 
