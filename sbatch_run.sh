#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J compass

# priority
# Unused because assumption is that this script is called via the batch_data_generation.sh script
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output ../sbatch_output/slurm/compass_run_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH -c 12
#SBATCH -N 1

# Unused because assumption is that this script is called via the batch_data_generation.sh script
##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=1-75
# --------------------------------------------------------------------------------------

# BASIC SETUP

# Read in arguments:
input_file=None
output_folder=/users/afengler/data/proj_compass_ddm/sbatch_output/
criterion=None
id=None
multiprocess=1
show_plots=0
while [ ! $# -eq 0 ]
    do
        case "$1" in
            --input_file | -if)
                input_file=$2
                ;;
            --output_folder | -of)
                output_folder=$2
                ;;
            --criterion | -c)
                criterion=$2
                ;;
        esac
        shift 2
    done

# USER-INPUT NEEDED
source /users/afengler/.bashrc  # NOTE: This file needs to contain conda initialization stuff

# TODO: This double conda deactivate can be simplified further --> key is understanding how to handle .bashrc / .bash_profile correctly
conda deactivate
conda deactivate
conda activate compass_ddm

echo "The input file passed is: $input_file"
echo "The output folder passed is: $output_folder"
echo "The criterion passed is: $criterion"

if [ -z ${SLURM_ARRAY_TASK_ID} ];
then
    python -u PowerAnalysis.py --input_file $input_file \
                               --output_folder $output_folder \
                               --criterion $criterion \
                               --multiprocess $multiprocess \
                               --show_plots $show_plots \
                               
else
    python -u PowerAnalysis.py --input_file $input_file \
                               --output_folder $output_folder \
                               --criterion $criterion \
                               --multiprocess $multiprocess \
                               --show_plots $show_plots \
                               --id $SLURM_ARRAY_TASK_ID 
fi