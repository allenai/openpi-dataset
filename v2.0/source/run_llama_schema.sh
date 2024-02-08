#!/bin/bash

#SBATCH --job-name=llama-inference-7b
#SBATCH --nodelist=nlpgpu06
#SBATCH --gpus=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --open-mode append
#SBATCH -o llama-schema-30b-p1.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seacow@seas.upenn.edu
#SBATCH --time=12:59:59

python predict_schema.py --model llama-30b --prompt 1
