#!/bin/bash
#SBATCH --job-name=scdnn
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cluster-node6
#SBATCH --mem 24G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_cnn_final2.txt
#SBATCH --error=job_error_cnn_final2.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp.py --task_num 15 --patience 100
