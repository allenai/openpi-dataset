#!/bin/bash

#SBATCH --job-name=llama-state-65b
#SBATCH --nodelist=nlpgpu06
#SBATCH --gpus=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=400GB
#SBATCH --open-mode append
#SBATCH -o llama-state-65b.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seacow@seas.upenn.edu
#SBATCH --time=12:00:00

python predict_states.py --model llama-65b --device_map ./llama-65b-device-map.json
