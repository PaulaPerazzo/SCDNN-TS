#!/bin/bash
#SBATCH --job-name=scdnn_dil
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem 24G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_cnn_dilatada_alation_3blocks.txt
#SBATCH --error=job_error_cnn_dilatada_alation_3blocks.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp_dil.py --task_num 22 --lr 0.0001 --weight_decay 0.0001 --patience 100
