#!/bin/bash
#backup patch models data
. backup.bash
git reset --hard
git pull
. clear_all_generated_data.bash
cd ../

module load lang/Python
module load numlib/cuDNN

python3 -m venv .venv
pip install --upgrade pip
cp ../requirements.txt .
pip install -r requirements.txt

chmod +x scripts/hpc_scripts/*.bash

for f in scripts/hpc_scripts/scrip2*.bash; do
    sbatch "$f"
done

