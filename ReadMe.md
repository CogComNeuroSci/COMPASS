[![Uni](https://img.shields.io/badge/University-Ghent%20University-brightgreen)](https://img.shields.io/badge/University-Ghent%20University-brightgreen)
[![Date](https://img.shields.io/badge/Last%20update-2023-yellow)](https://img.shields.io/badge/Last%20update-2023-yellow)

# Computational Power Analysis using Simulations "COMPASS" toolbox on DDM

Expanding COMPASS toolbox to DDM models.

Additional to the original COMPASS, packages required by this extension for DDM: 

1.ssms: Python Package which collects simulators for Sequential Sampling Models.
For more details, see: https://github.com/AlexanderFengler/ssm-simulators

## The workflow of COMPASS DDM

Similar to the original COMPASS, COMPASS DDM follows these steps:

1. Sample true parameters from the distributions defined by a .csv input file, with ranges defined in ssms package.
2. Generate behavioral data by the true parameters sampled. Behavioral data includes both choices (denoted by 1 or -1) and reaction times.
3. Validate performance of generated behavioral data. If performance exceeds normal range, then go back to step 1
   Performance validation:
    A participant as well as its true parameter can be accepted only when the mean ACC is within 50% to 95% AND mean RT is within 0 to 10 s.
    Here mean ACC and mean RT of one participant are used as measure of performance. For one choice, ACC is defined by the sign of drift parameter and the sign of that choice. For example, if the option denoted by -1 is chosen and the drift parameter is also a negative value, then it is a correct choice. If the drift parameter equals to zero, the parameters will be resampled.
5. Estimate the best fitting parameters for each participant given the simulated behavioral data.
6. Compute statistics on one sample and compare to the cut-off. Statistics including:
  Internal_correlation: correlation coefficients between sampled and estimated parameter values.
  External_correlation: correlation coefficients between an external measurement (which is assumed to be Gaussian distributed) and estimated parameter values.
  Groupdifference: t-values measuring a group difference between estimated parameters of two groups
7. Repeat sampling and evaluate the proportion of statistics reached the cut-off value

## Steps to compute power of DDM
###  1.Define the distributions from which true parameters are sampled
Creat a csv file, with name of "InputFile_IC_DDM", "InputFile_EC_DDM", or "InputFile_GD_DDM" corresponding to each criterion.

For IC criterion, you should define:
  model: index of DDM model which should be matched with ssms package
  ntrials: number of trials that will be used to do the parameter recovery analysis for each participant.
  npp: integer, number of participants in the study.
  “mean_{}”s: means of true parameter distribution. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: mean_v, mean_a, mean_z, mean_t
  “std_{}”s: stds of true parameter distribution. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: std_v, std_a, std_z, std_t
  tau: the cut-off value of correlation coef
  nreps: number of samples to calculate proportion(probability) of which statistics exceed cut-off value
  full_speed: defines whether multiple cores on the computer will be used in order to estimate the power.
  output_folder: path to save results

For EC criterion, you should define:
  model: index of DDM model which should be matched with ssms package
  ntrials: number of trials that will be used to do the parameter recovery analysis for each participant.
  npp: integer, number of participants in the study.
  “mean_{}”s: means of true parameter distribution. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: mean_v, mean_a, mean_z, mean_t
  “std_{}”s: stds of true parameter distribution. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: std_v, std_a, std_z, std_t
  par_ind: parameter index of the parameter of interest, according to the order from ssms. START FROM 0. E.g., par_ind = 0, corresponding to "v"
  True_correlation: the hypothesized correlation between the learning rate and the external measure theta.   
  TypeIerror: critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
  nreps: number of samples to calculate proportion(probability) of which statistics exceed cut-off value
  full_speed: defines whether multiple cores on the computer will be used in order to estimate the power.
  output_folder: path to save results

For GD criterion, you should define:
  model: index of DDM model which should be matched with ssms package
  ntrials: number of trials that will be used to do the parameter recovery analysis for each participant.
  npp: integer, number of participants in the study.
  “mean_{}”s: means of true parameter distribution. There should be TWO values separated by COMMA in the cell corresponding to the parameter you want to compare. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: mean_v, mean_a, mean_z, mean_t
  “std_{}”s: stds of true parameter distribution. There should be TWO values separated by COMMA in the cell corresponding to the parameter you want to compare. The order of parameters MUST BE MATCH with that in ssms. e.g., for ddm model, the order must be: stds_v, stds_a, stds_z, stds_t
  par_ind: parameter index of the parameter of interest, according to the order from ssms. START FROM 0. E.g., par_ind = 0, corresponding to "v"
  True_correlation: the hypothesized correlation between the learning rate and the external measure theta.   
  TypeIerror: critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
  nreps: number of samples to calculate proportion(probability) of which statistics exceed cut-off value
  full_speed: defines whether multiple cores on the computer will be used in order to estimate the power.

### 2. run COMPASS
1. Open Anaconda prompt, type:conda activate pyPower
2. run: python PowerAnalysis.py IC_DDM, python PowerAnalysis.py GD_DDM, python PowerAnalysis.py EC_DDM depending on the criterion that you want to use.


