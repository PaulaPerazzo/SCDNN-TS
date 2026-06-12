#!/bin/bash
#SBATCH --job-name=scdnn_dil
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_test_cnn_dilatada_opt1.txt
#SBATCH --error=job_error_test_cnn_dilatada_opt1.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp_dil.py --epochs 100 --task_num 3
