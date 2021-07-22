#!/bin/bash

#name job pythondemo, output to slurm file, use partition all, run for 1500 minutes and use 40GB of ram
#SBATCH -J 'ES_mask_data'
#SBATCH -o logfiles/ES_mask_data-%j.out
#SBATCH --error=logfiles/ES_mask_data%j.err
#SBATCH -p all
#SBATCH -t 1000
#SBATCH -c 20 --mem 70000
#SBATCH --mail-type ALL
#SBATCH --mail-user pnaphade@princeton.edu
#SBATCH --array=0-22


module load pyger
export PYTHONMALLOC=debug

python -duv /tigress/pnaphade/Eternal_Sunshine/scripts/mask_data.py

