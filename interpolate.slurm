#!/bin/bash
#BATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --job-name=intl25
#SBATCH --gres=gpu
#SBATCH --qos=zmfm97-qos
#SBATCH -t 02-00
#SBATCH --mem=38g
#SBATCH --nodelist=gpu5

source /etc/profile
source ../bin/activate
module load cuda/10.0-cudnn7.4

python interpolate.py


