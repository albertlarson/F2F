#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=1
#SBATCH --mem=130000
#SBATCH -o 999999_f2f-%j.out  # %j = job ID

module purge

eval "$(conda shell.bash hook)"

conda info --envs

conda activate /work/albertl_uri_edu/.conda/envs/sm

python -u 9999_f2f.py  > 9999999_f2f.out 