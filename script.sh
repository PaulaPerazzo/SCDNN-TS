#!/bin/bash
#SBATCH --job-name=scdnn
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 24G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH -o job.log
#SBATCH --output=job_output_cnn_bench_ablation_3blocks.txt
#SBATCH --error=job_error_cnn_bench_ablation_3blocks.txt

source env/bin/activate

# python3 generate_data.py
python3 train_temp.py --task_num 23 --patience 100
