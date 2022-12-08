#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=nldas_summing
#SBATCH --mem=16000
#SBATCH -o job-%j.out  # %j = job ID

module load python/3.9.1
module load miniconda
source activate sm

python -u nldas_summing.py  > jobpy_summing.out 
