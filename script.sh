#!/bin/bash
#SBATCH --job-name=scdnn_dil
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_4.txt
#SBATCH --error=job_error_4.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp.py
