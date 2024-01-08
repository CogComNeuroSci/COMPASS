#!/bin/bash

# Set parameters
criterion=EC
input_file="ClusterInput/InputFile_"$criterion"_DDM.csv"
n_rows_input_file=75
output_folder="/users/afengler/data/proj_compass_ddm/sbatch_output/data_"$criterion

# Run sbatch script
# sbatch -p batch --account=carney-frankmj-condo --array=0-$n_rows_input_file sbatch_run.sh \
#             --input_file $input_file \
#             --output_folder $output_folder \
#             --criterion $criterion | cut -f 4 -d' '

sbatch -p batch --array=0-$n_rows_input_file sbatch_run.sh \
            --input_file $input_file \
            --output_folder $output_folder \
            --criterion $criterion | cut -f 4 -d' '