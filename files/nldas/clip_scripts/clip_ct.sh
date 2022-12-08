#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --job-name=nldas_summing
#SBATCH --mem=32000
#SBATCH -o job-%j.out  # %j = job ID

module load python/3.9.1
module load miniconda
source activate sm

python -u nldas_clip_ct.py  > nldas_clip_ct.out 
