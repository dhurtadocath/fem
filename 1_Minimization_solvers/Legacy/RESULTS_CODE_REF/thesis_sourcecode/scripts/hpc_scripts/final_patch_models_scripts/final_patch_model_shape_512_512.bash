#!/bin/bash -l
#SBATCH --mail-user=boghos.youseef.001@student.uni.lu
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --qos normal
#SBATCH --time=23:59:00
#SBATCH -p gpu
#SBATCH --mem=20GB


python3 main.py train -mt p -s 512 512 -e 100 -n final_patch_model -ds 0 -v

