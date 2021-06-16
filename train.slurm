#!/bin/bash 
#SBATCH -N 1 
#SBATCH -c 8 
#SBATCH -p res-gpu-large 
#SBATCH --job-name=UKT1 
#SBATCH --gres=gpu 
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00 
#SBATCH --mem=28g

source /etc/profile 
source ../bin/activate 
module load cuda/10.0-cudnn7.4 

python train.py --config_file UKPNTest1.json


