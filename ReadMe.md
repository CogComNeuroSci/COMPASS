[![Uni](https://img.shields.io/badge/University-Ghent%20University-brightgreen)](https://img.shields.io/badge/University-Ghent%20University-brightgreen)
[![Date](https://img.shields.io/badge/Last%20update-2022-yellow)](https://img.shields.io/badge/Last%20update-2022-yellow)

# COmputational Power Analysis using Simulations "COMPASS" toolbox 

This toolbox has been developed to evaluate statistical power when using parameter estimates from computational models.

In the current version, use is limited to the Rescorla-Wagner (RW) model in two-armed bandit tasks.

More details can be found in the manuscript: *Insert link when preprint is online*

## Installation guide
*Step 1:* Downloading all code and storing them locally on your own pc. 

*Step 2:* Creating the PyPower environment  
  
  Normally, power analyses with COMPASS should be possible in a basic Python environment.  
  Nevertheless, to control for version issues, we provide environment files for windows and mac users.  
   * Install Anaconda 3 by following their [installation guide](https://docs.anaconda.com/anaconda/install/windows/)  
   * When the installation is complete, open an Anaconda prompt  
   * Go to the directory where the COMPASS files are stored using ```cd```  
   * For windows, run: ```conda env create --file environment_windows.yml```   and for mac, run ```conda env create --file environment_mac.yml```  
   * Allow the installation of all required packages  
   
## The model and task currently implemented in COMPASS
### The RW model

```To specify by the user: meanLR, sdLR, meanInverseTemperature and sdInverseTemperature```

The RW model is used to fit participants’ behaviour in this task.   
The core of the model is formed by the delta-learning rule and the softmax choice rule.   
The model has two free parameters: the learning rate and the inverse temperature (see manuscript for details).    
The population distribution of these parameters can be specified by the user in the csv files by completing *meanLR, sdLR, meanInverseTemperature* and *sdInverseTemperature*.  

### Two-armed bandit task

```To specify by the user: ntrials, nreversals, reward_probability``` 

Based on the parameters that are implemented in the csv files, a task design is created.  
In this task design, there are two stimuli/bandits and two possible actions (selecting bandit 1 or 2).   
One bandit is more optimal since it has a higher probability of reward (specified by *reward_probability*).   
For simplicity, we implement that Pr(Reward | optimal_bandit) = 1- Pr(Reward | suboptimal_bandit).  
As in classic reversal learning tasks, we provide the option that the identity of the optimal bandit can reverse. Here, one has to specify the frequency of rule reversals (*nreversals*). If set to zero, there are no reversals.   
The design is created for *ntrials*. As demonstrated in the manuscript, this is a crucial variable for obtaining reliable parameter estimates and high statistical power.  

## Important note: the required computational time. 

```To specify by the user: npp, nreps, full_speed```

As we perform parameter estimations for *nreps* Monte Carlo repetitions, computational time can increase exponentially.  
The computational time strongly depends on the number of participants (*npp*) and the number of Monte Carlo repetitions (*nreps*). We recommend to set *nreps* to 250. Smaller numbers can be used as well but then power computations will be less precise.   
Notably, also increasing the number of trials in the task design (*ntrials*) can significantly increase the power computation time in COMPASS.  

As a partial solution for this computation time, the option is included to run the power analysis on multiple cores. This happens when the user defines the *full_speed* option as 1; if this option is activated, all minus two cores on the computer will be used for power computations.  

When running COMPASS it will asap provide an estimate of how long it will take to calculate the power for each power computation (each row of the csv files specify one power computation).  
This estimate is based on the time it takes to execute a single repetition and calculated by multiplying the total number of repetitions by the time required for a single repetition, divided by the number of cores that are used in the power analysis.   

If you want to stop the process whilst running, you can use 'ctrl + C' in the anaconda prompt shell. This will stop the execution of the script.   

## Runnig power computations with COMPASS  
As described in the manuscript, three criteria for power computations are specified.  
For each criterion (IC, EC or GD), we provide a csv file which holds the power variables that should be specified by the user.  
For all criteria, power is specified as  

  ```power = Pr(Statistic > cut-off | Hypothesis)```

Here, the statistic differs across criteria and the cut-off and hypothesis should be specified by the user.  

Power computations consist of the following five steps:   
  1. Sample *npp* participants from the population.   
  This sampling process is guided by the hypothesis that is specified by the user in the csv files.  (population distribution of parameter values (for IC), true correlation (for EC) or difference between groups (for GD))   
  2. Simulate data for each participant.  
  3. Estimate the best fitting parameters for each participant given the simulated data.   
  These are the ‘estimated parameters’.  
  4. Compute statistics.  
  The statistic differs across criteria.    
      - _internal_correlation_: correlation between sampled and estimated parameter values.    
      - _external_correlation_: correlation between estimated parameter values and external measure (e.g., questionnaire score).    
      - _group_difference_: T-statistic of difference in parameter values between two groups.    
  5. Evaluate which proportion of statistics reached the cut-off value.  

### The steps for the user
1. Make sure that COMPASS is installed correctly (see Installation guide above).  

2. Choose a criterion and specify variables in the corresponding csv file.  
*Notice that multiple rows can be specified in the csv files, power computations will be performed for each row that is completed by the user*    
  
  2a) Internal Correlation (IC): Correlation between sampled and estimated parameter values.  
  
  Open the InputFile_IC and specify  
  * _ntrials_: integer 𝜖 [5, +∞[
	**number of trials within the experiment (minimal 5)**
  * _nreversals_: integer 𝜖 [0, ntrials[
	**number of rule reversals within the experiment**
  * _npp_: integer 𝜖 [5, +∞[ 
	**total number of participants within the experiment (minimal 5)**
  * _meanLR_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of learning rates**
  * _sdLR_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of learning rates**
  * _meanInverseTemperature_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of inverse temperatures**
  * _sdInverseTemperature_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of inverse temperatures**
  * _reward_probability_: float 𝜖 [0, 1] 
	**The probability of reward for the optimal bandit in the two-arm bandit task.**
  * _tau_: float 𝜖 [0, 1] 
	**the value against which the obtained statistic will be compared to define significance of the repetition.**
	- correlation: cut_off = minimally desired correlation - recommended: 0.75
  * _full_speed_: integer (0 or 1)
	**Define whether you want to do the power analysis at full speed.**
	- 0 = only one core will be used (slow)
	- 1 = (all-2) cores will be used (much faster, recommended unless you need your computer for other intensive tasks such as meetings)
  * _nreps_: integer 𝜖 [1, +∞[ 
	**Number of repetitions that will be conducted to estimate the power**
	- Recommended number: 250
  * _output_folder_: string
	**Path to the folder where the output-figure(s) will be stored**
	- e.g. "C:\Users\maudb\Downloads"

  2b) External Correlation (EC): Correlation between estimated parameter values and external measure (e.g., Questionnaire scores).  
  
  Open the InputFile_EC and specify  
  * _ntrials_: integer 𝜖 [5, +∞[
	**number of trials within the experiment (minimal 5)**
  * _nreversals_: integer 𝜖 [0, ntrials[
	**number of rule reversals within the experiment**
  * _npp_: integer 𝜖 [5, +∞[ 
	**total number of participants within the experiment (minimal 5)**
  * _meanLR_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of learning rates**
  * _sdLR_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of learning rates**
  * _meanInverseTemperature_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of inverse temperatures**
  * _sdInverseTemperature_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of inverse temperatures**
  * _reward_probability_: float 𝜖 [0, 1] 
	**The probability of reward for the optimal bandit in the two-arm bandit task.**
  *_True_correlation_: float 𝜖 [-1, 1] 
  **The hypothesized correlation between the learning rate parameter values and the external measure**
  * _TypeIerror_: float 𝜖 [0, 1] 
	**The allowed probability to make a type I error; the significance level**
	- standard (and recommended) value: 0.05
	- correlation: cut_off = minimally desired correlation - recommended: 0.75
  * _full_speed_: integer (0 or 1)
	**Define whether you want to do the power analysis at full speed.**
	- 0 = only one core will be used (slow)
	- 1 = (all-2) cores will be used (much faster, recommended unless you need your computer for other intensive tasks such as meetings)
  * _nreps_: integer 𝜖 [1, +∞[ 
	**Number of repetitions that will be conducted to estimate the power**
	- Recommended number: 250
  * _output_folder_: string
	**Path to the folder where the output-figure(s) will be stored**
	- e.g. "C:\Users\maudb\Downloads"
	
  2c) Group Difference (GD): T-statistic for difference between estimated parameter values of two groups.

  Open the InputFile_GD and specify  
  * _ntrials_: integer 𝜖 [5, +∞[
	**number of trials within the experiment (minimal 5)**
  * _nreversals_: integer 𝜖 [0, ntrials[
	**number of rule reversals within the experiment**
  * _npp_group_: integer 𝜖 [5, +∞[ 
	**number of participants within the experiment (minimal 5)**
  * _meanLR_g1_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of learning rates for group 1**
  * _sdLR_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of learning rates for group 1** 
  * _meanLR_g2_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of learning rates for group 2**
  * _sdLR_g2_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of learning rates for group 2** 
  * _meanInverseTemperature_g1_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of inverse temperatures for group 1**
  * _sdInverseTemperature_g1_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of inverse temperatures for group 1**
  * _meanInverseTemperature_g2_: float 𝜖 [0, 1]
	**mean of the assumed population distribution of inverse temperatures for group 2**
  * _sdInverseTemperature_g2_: float 𝜖 [0, 1]
	**sd of the assumed population distribution of inverse temperatures for group 2**
  * _reward_probability_: float 𝜖 [0, 1] 
  **The probability of reward for the optimal bandit in the two-arm bandit task.**
  * _TypeIerror_: float 𝜖 [0, 1] 
	**The allowed probability to make a type I error; the significance level**
	- standard (and recommended) value: 0.05
  * _full_speed_: integer (0 or 1)
	**Define whether you want to do the power analysis at full speed.**
	- 0 = only one core will be used (slow)
	- 1 = (all-2) cores will be used (much faster, recommended unless you need your computer for other intensive tasks such as meetings)
  * _nreps_: integer 𝜖 [1, +∞[ 
	**Number of repetitions that will be conducted to estimate the power**
	- Recommended number: 250
  * _output_folder_: string
	**Path to the folder where the output-figure(s) will be stored**
	- e.g. "C:\Users\maudb\Downloads"
  
3. Run the PowerAnalysis.py script using the correct Anaconda 3 environment.   
   
   If one followed the Installation guide above, a PyPower environment has been created.  
   
   To use this environment:   
   * Open Anaconda prompt
   * Now, run: ```conda activate pyPower```
   
   To run COMPASS:
   * Go to the directory where the COMPASS files are stored using ```cd```  
   * Now, run: ```python PowerAnalysis.py IC```, ```python PowerAnalysis.py EC``` or ```python PowerAnalysis.py GD``` depending on the criterion that you want to use.

4. Check the output in the shell & the stored figure(s) in the _output_folder_  
   * _power estimate_: the probability to obtain adequate parameter estimates.  
   * _probability density plot of the Statistic of interest_: a plot visualising the obtained values for the Statistic of interest in all power recovery analyses
     - x-axis: values for the statistic of interest (correlation or T-Statistic)
     - y-axis: probability density for each value

    Example output (EC criterion): 
    
    <img width="800" alt="image" src="https://github.com/CogComNeuroSci/COMPASS/blob/main/Figures_ReadMe/Output_terminal.png">  
    
    <img width="500" alt="image" src="https://github.com/CogComNeuroSci/COMPASS/blob/main/Figures_ReadMe//Output_plot.png">
    
# Contact
- Corresponding author: Pieter Verbeke
    * [E-mail me at pjverbek (dot) Verbeke (at) UGent (dot) be](mailto:pjverbek.Verbeke@UGent.be)
- First author (internship student): Maud Beeckmans 
    * [E-mail me at Maud (dot) Beeckmans (at) UGent (dot) be](mailto:Maud.Beeckmans@UGent.be)
- Supervising PhD researcher: Pieter Huycke
    * [E-mail me at Pieter (dot) Huycke (at) UGent (dot) be](mailto:Pieter.Huycke@UGent.be)
- Supervising PI: Tom Verguts
    * [E-mail me at Tom (dot) Verguts (at) UGent (dot) be](mailto:Tom.Verguts@UGent.be)

**Last edit: December 20th 2022**
