#!/bin/bash -l
#SBATCH --mail-user=boghos.youseef.001@student.uni.lu
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --qos normal
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH --mem=5GB


python3 main.py train -mt sp -s 128 -e 100 -n surface_points_model_patch_25 -ds 0