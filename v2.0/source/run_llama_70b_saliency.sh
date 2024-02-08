#!/bin/bash

#SBATCH --job-name=llama_7b_ft
#SBATCH --nodelist=nlpgpu07
#SBATCH --gpus=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=250GB
#SBATCH --open-mode append
#SBATCH -o llama_70b.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hainiu.xu@kcl.ac.uk
#
#
python predict_salience_llama.py
