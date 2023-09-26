[![Uni](https://img.shields.io/badge/University-Ghent%20University-brightgreen)](https://img.shields.io/badge/University-Ghent%20University-brightgreen)
[![Date](https://img.shields.io/badge/Last%20update-2023-yellow)](https://img.shields.io/badge/Last%20update-2023-yellow)

# COmputational Power Analysis using Simulations "COMPASS" toolbox on DDM

Expanding COMPASS toolbox to DDM models.

Additional to the original COMPASS, packages required by this extension for DDM: 

1.ssms: Python Package which collects simulators for Sequential Sampling Models.
For more details, see: https://github.com/AlexanderFengler/ssm-simulators

## The workflow of COMPASS DDM

Similar to the original COMPASS, COMPASS DDM follows these steps:

1. Sample true parameters from the distributions you defined within the parameter bounds defined in ssms package.
2. Generate behavioral data by the true parameters sampled. Behavioral data includes both choices (denoted by 1 or -1) and reaction times.
3. Validate performance of generated behavioral data. If performance exceeds normal range, then go back to step 1
4. Estimate the best fitting parameters for each participant given the simulated data.
5. Compute statistics.
internal_correlation: correlation between sampled and estimated parameter values.
6. Evaluate which proportion of statistics reached the cut-off value.

## Steps to compute power of DDM
###  1.Define the distributions from which true parameters are sampled

To make it suitable for real behavioral data, parameters sampled to compute power should generate performance similar to real participants. Performance heatmaps of different combinations of parameters are plotted helping to choose ranges of true parameters.

Here mean ACC and mean RT of one participant are used as measure of performance. For one choice, ACC is defined by the sign of drift parameter and the sign of that choice. For example, if the option denoted by -1 is chosen and the drift parameter is also a negative value, then it is a correct choice. If the drift parameter equals to zero, the parameters will be resampled.

A participant as well as its true parameter can be accepted only when the mean ACC is within 50% to 95% AND mean RT is within 0 to 10 s.

HOW TO DO: set "parameter_range = 1" in the "test.py" and run the file 

### 2. try parameter recovery on one participant

Before computing power analysis on large size of samples, let's start from recovering parameters on one participant.

HOW TO DO: set "parameter_recovery = 1" in the "test.py" and run the file 

### 3. set input file for power analysis

You can specify multiple rows in your input file.

For IC criterion, open InputFile_IC.csv and specify:
  
  * model: string, the DDM model of interest. model must be included by ssms package.
  * ntrials: integer, the number of trials
  * npp: integer, the number of participants
  * mean_{}: float, means of parameters corresponding to the model type
    NOTE: the ORDER of parameters must be the same as they are ordered in the ssms package.
    NOTE: to see the order of parameters, use: ssms.config.model_config[model]['params']
  * std_{}: float, stds of parameters corresponding to the model type
    NOTE: the ORDER of stds must correspond to the means
  * tau: float 𝜖 [0, 1] the value against which the obtained statistic will be compared to define significance of the repetition
  * nreps: integer 𝜖 [1, +∞] Number of repetitions that will be conducted to estimate the power
  * full_speed: integer (0 or 1) Define whether you want to do the power analysis at full speed.
    0 = only one core will be used (slow)
    1 = (all-2) cores will be used (much faster, recommended unless you need your computer for other intensive tasks such as meetings)
  * output_folder: string, path to the folder where the output-figure(s) will be stored
      

### 4. Run power computations for DDM

Run the PowerAnalysis.py script using the correct Anaconda 3 environment.

If one followed the Installation guide above, a PyPower environment has been created.

To use this environment:
  Open Anaconda prompt
  Now, run: conda activate pyPower
  
To run COMPASS:
  Go to the directory where the COMPASS files are stored using cd
  Now, run: python PowerAnalysis.py IC, python PowerAnalysis.py EC or python PowerAnalysis.py GD depending on the criterion that you want to use.

### 5. Check the output in the shell & the stored figure(s) in the output_folder

#### Power heatmap

After computing power on multiple combinations of npp and ntrials, you can get a power heatmap to choose the optimal design of your research.

HOW TO DO: 
1. Specify the following variables of results in plot.py
* ResultPath：string, the path you gather results of power analysis
* DDM_id: string, correspond to the model in the input file, e.g., "ddm"
* tau: float, correspond to the tau in the input file
* nreps: integer 𝜖 [1, +∞] Number of repetitions that will be conducted to estimate the power
* range_ntrials: list, list of trials of interest
* range_npp: list, list of trials of interest
* p_list: list, list of parameters

2. Set "plot_heatmap = 1"
3. Run the file

#### Distribution of statistics of one computation 

HOW TO DO:
1. Specify the "range_ntrials" and "range_npp" as the setting of computation of interest in plot.py
   e.g.,
     range_ntrials = [60]
     range_npp = [20]
2. Set "plot_single_setting = 1"
3. Run the file
