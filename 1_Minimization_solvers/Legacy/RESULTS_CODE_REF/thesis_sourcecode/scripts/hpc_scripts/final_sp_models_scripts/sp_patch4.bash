#!/bin/bash -l
 #SBATCH --mail-user=boghos.youseef.001@student.uni.lu
 #SBATCH --mail-type=ALL
 #SBATCH -N 1
 #SBATCH --ntasks-per-node=1
 #SBATCH -c 1
 #SBATCH -G 1
 #SBATCH --qos normal
 #SBATCH --time=2:00:00
 #SBATCH -p gpu
 #SBATCH --mem=10GB


 python3 main.py train -mt sp -p 4 -s 512 512 -e 1000 -n surface_points_model_patch_4 -ds 0

