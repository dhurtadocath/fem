#!/bin/bash -l
#SBATCH --mail-user=boghos.youseef.001@student.uni.lu
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --qos normal
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --mem=10GB


python3 main.py train -mt p -s 1024 1024 -e 200000 -n patch_model_full_data_set_regularizer_sample_weights -ds 0 -w -r