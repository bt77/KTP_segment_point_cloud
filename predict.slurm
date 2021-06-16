#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --job-name=pUK0
#SBATCH --gres=gpu
#SBATCH --qos=zmfm97-qos
#SBATCH -t 02-00
#SBATCH --mem=58g
#SBATCH --nodelist=gpu6

source /etc/profile
source ../bin/activate
module load cuda/10.0-cudnn7.4

python predict.py --config_file UKPNTest0.json --ckpt log/UKPNTest0/best_model_epoch_040.ckpt \
	                  --set=validation \
			                    --num_samples=100

