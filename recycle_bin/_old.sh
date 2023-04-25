#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=f2f_9
#SBATCH --mem=130000
#SBATCH -o run-%j.out  # %j = job ID


# eval "$(conda shell.bash hook)"
# source /work/albertl_uri_edu/fluxtoflow/files_for_paper/f2f_venv/bin/activate
# python3 --version
# which python3

# conda info --envs

# source activate /work/albertl_uri_edu/.conda/envs/f2f_2/bin/activate

python3 --version
which python3

conda info --envs

python3 -u run.py  > run.out 