#!/bin/bash
#SBATCH --job-name=scdnn
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_2.txt
#SBATCH --error=job_error_2.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp.py --epochs 100 --task_num 1
