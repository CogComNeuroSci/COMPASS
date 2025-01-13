#!/bin/bash

# BASIC SETUP
# Read in arguments:
input_file=None
output_folder=/users/afengler/data/proj_compass_ddm/sbatch_output/
criterion=None
id=None
multiprocess=1
show_plots=0
optimalgo=0
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
            --optimalgo | -o)
                optimalgo=$2
                ;;
            --show_plots | -s)
                show_plots=$2
                ;;
            --multiprocess | -mp)
                multiprocess=$2
                ;;
        esac
        shift 2
    done

python -u PowerAnalysis.py --input_file $input_file \
                            --output_folder $output_folder \
                            --criterion $criterion \
                            --multiprocess $multiprocess \
                            --show_plots $show_plots \
                            --id $id \