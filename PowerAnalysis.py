# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:04:23 2021
MOdified on june 2023

@author: maudb, Luning He
"""
import os
os.chdir('D:horiz/IMPORTANT/0study_graduate/Pro_COMPASS/COMPASS_DDM/')
#%%
HPC = False

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Functions_DDM import Incorrelation_repetition, groupdifference_repetition, Excorrelation_repetition
from scipy import stats as stat
from datetime import datetime
import ssms
if HPC == False:
    import seaborn as sns
    import matplotlib.pyplot as plt

#This is to avoid warnings being printed to the terminal window
import warnings
warnings.filterwarnings('ignore')

#%% RL functions

def power_estimation_Incorrelation_RL(npp = 30, ntrials = 480, nreversals = 12, cut_off = 0.7, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LRdistribution = 0.5, SD_LRdistribution = 0.1,
                                 mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0):
    
    
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistribution: float
        Mean for the normal distribution to sample learning rates from.
    SD_LRdistribution: float
        Standard deviation for the normal distribution to sample learning rates from.
    mean_inverseTempdistribution: float
        Mean for the normal distribution to sample inverse temperatures from.
    SD_inverseTempdistribution: float
        Standard deviation for the normal distribution to sample inverse temperatures from.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the correlation value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if their correlation with the true parameters is minimum the cut_off.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    #divide process over multiple cores
    pool = Pool(processes = n_cpu)
    LR_distribution = np.array([mean_LRdistribution, SD_LRdistribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])
    out = pool.starmap(Incorrelation_repetition, [(inverseTemp_distribution, LR_distribution, npp, ntrials,
                                                 start_design, rep, nreps, n_cpu) for rep in range(nreps)])
    pool.close()
    pool.join()

    allreps_output = pd.DataFrame(out, columns = ['correlations'])


    power_estimate = np.mean((allreps_output['correlations'] >= cut_off)*1)
    print(str("\nProbability to obtain a correlation(true_param, param_estim) >= {}".format(cut_off)
          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate*100)))

    return allreps_output, power_estimate


def power_estimation_Excorrelation_RL(npp = 100, ntrials = 480, nreversals = 12, typeIerror = 0.05, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LRdistribution = 0.5, SD_LRdistribution = 0.1,
                                 mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0, True_correlation = .5):
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    typeIerror : float
        Critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistribution: float
        Mean for the normal distribution to sample learning rates from.
    SD_LRdistribution: float
        Standard deviation for the normal distribution to sample learning rates from.
    mean_inverseTempdistribution: float
        Mean for the normal distribution to sample inverse temperatures from.
    SD_inverseTempdistribution: float
        Standard deviation for the normal distribution to sample inverse temperatures from.
    True_correlation: float
        The hypothesized correlation between the learning rate and the external measure theta.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the correlation value on each repetition, the p-value and the p-value if estimates would be perfect.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain significant correlations with external measures.
    Parameter estimates are considered to be adequate if correctly reveal a significant correlation when a significant correlation.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    #Use beta_distribution to determine the p-value for the hypothesized correlation
    beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
    true_pValue = 1-beta_distribution.cdf(True_correlation)
    tau = -beta_distribution.ppf(typeIerror/2)
    
    #compute conventional power
    noncentral_beta = stat.beta((npp/2)-1, (npp/2)-1, loc = -1+True_correlation, scale = 2)
    conventional_power = 1-noncentral_beta.cdf(tau)

    print(str("\nThe correlation cut-off value is: {}".format(np.round(tau,2))))
    print(str("\np-value for true correlation is :{}".format(np.round(true_pValue,5))))
    print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(conventional_power*100,2))))

    #divide process over multiple cores
    pool = Pool(processes = n_cpu)
    LR_distribution = np.array([mean_LRdistribution, SD_LRdistribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])
    out = pool.starmap(Excorrelation_repetition, [(inverseTemp_distribution, LR_distribution, True_correlation, npp, ntrials,
                                                 start_design, rep, nreps, n_cpu) for rep in range(nreps)])
    pool.close()
    pool.join()

    allreps_output = pd.DataFrame(out, columns = ['Statistic','estimated_pValue', 'True_pValue'])

    #Compute power if estimates would be perfect.
    #power_true = np.mean((allreps_output['True_pValue'] <= typeIerror/2)*1)
    #print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(power_true*100,2))))

    #Compute power for correlation with estimated parameter values.
    power_estimate = np.mean((allreps_output['estimated_pValue'] <= typeIerror/2)*1)
    print(str("\nProbability to obtain a significant correlation between model parameter and an external measure that is {} correlated".format(True_correlation)
          + " with {} trials and {} participants: {}%".format(ntrials, npp, np.round(power_estimate*100,2))))

    return allreps_output, power_estimate

def power_estimation_groupdifference_RL(npp_per_group = 20, ntrials = 480, nreps = 100, typeIerror = 0.05,
                                     high_performance = False, nreversals = 12, reward_probability = 0.8,
                                     mean_LRdistributionG1 = 0.5, SD_LRdistributionG1 = 0.1,
                                     mean_LRdistributionG2 = 0.5, SD_LRdistributionG2 = 0.1,
                                     mean_inverseTempdistributionG1 = 2.0, SD_inverseTempdistributionG1 = 1.0,
                                     mean_inverseTempdistributionG2 = 2.0, SD_inverseTempdistributionG2 = 1.0):
    """
    Parameters
    ----------
    npp_per_group : integer
        Number of participants per group in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    typeIerror : float
        Critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistributionG1: float
        Mean for the normal distribution to sample learning rates for group 1.
    SD_LRdistributioG1: float
        Standard deviation for the normal distribution to sample learning rates for group 1.
    mean_inverseTempdistributionG1: float
        Mean for the normal distribution to sample inverse temperatures for group 1.
    SD_inverseTempdistributionG1: float
        Standard deviation for the normal distribution to sample inverse temperatures for group 1.
    mean_LRdistributionG2: float
        Mean for the normal distribution to sample learning rates for group 2.
    SD_LRdistributioG2: float
        Standard deviation for the normal distribution to sample learning rates for group 2.
    mean_inverseTempdistributionG2: float
        Mean for the normal distribution to sample inverse temperatures for group 2.
    SD_inverseTempdistributionG2: float
        Standard deviation for the normal distribution to sample inverse temperatures for group 2.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the p-value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which a significant group difference was found divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if they correctly reveal the group difference when a true group difference of size 'cohens_d' exists.
    Power is calculated using a Monte Carlo simulation-based approach.
    """

    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    if high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1
    if __name__ == '__main__':
        
        #Use t_distribution to determine the p-value for the hypothesized cohen's d
        true_pValue = 1-stat.t.cdf(cohens_d*np.sqrt(npp_per_group), (npp_per_group-1)*2)
        tau = -stat.t.ppf(typeIerror/2, (npp_per_group-1)*2)
        
        #Compute conventional power
        conventional_power = 1-stat.nct.cdf(tau, (npp_per_group-1)*2, cohens_d*np.sqrt(npp_per_group))

        print(str("\nThe t-distribution cut-off value is: {}".format(np.round(tau,2))))
        print(str("\np-value for given cohen's d is :{}".format(np.round(true_pValue,5))))
        print("\nProbability to obtain a significant group difference under conventional power implementation: {}%".format(np.round(conventional_power*100,2)))

        #divide process over multiple cores
        if mean_LRdistributionG1 > mean_LRdistributionG2:
            LR_distributions = np.array([[mean_LRdistributionG1, SD_LRdistributionG1], [mean_LRdistributionG2, SD_LRdistributionG2]])
        else:
            LR_distributions = np.array([[mean_LRdistributionG2, SD_LRdistributionG2], [mean_LRdistributionG1, SD_LRdistributionG1]])

        inverseTemp_distributions = np.array([[mean_inverseTempdistributionG1, SD_inverseTempdistributionG1],
                                              [mean_inverseTempdistributionG2, SD_inverseTempdistributionG2]])
        pool = Pool(processes = n_cpu)
        out = pool.starmap(groupdifference_repetition, [(inverseTemp_distributions, LR_distributions, npp_per_group,
                                                     ntrials, start_design, rep, nreps, n_cpu, False) for rep in range(nreps)])
        # before calling pool.join(), should call pool.close() to indicate that there will be no new processing
        pool.close()
        pool.join()

        allreps_output = pd.DataFrame(out, columns = ['Statistic', 'PValue'])

        # check for which % of repetitions the group difference was significant
        # note that we're working with a one-sided t-test (if interested in two-sided need to divide the p-value obtained at each rep with 2)
        power_estimate = np.mean((allreps_output['PValue'] <= typeIerror/2))
        print(str("\nProbability to detect a significant group difference when the estimated effect size d = {}".format(np.round(cohens_d,3))
              + " with {} trials and {} participants per group: {}%".format(ntrials,
                                                                         npp_per_group, np.round(power_estimate*100,2))))
        return allreps_output, power_estimate

#%% DDM functions
def power_estimation_Incorrelation(npp = 30, ntrials = 480, cut_off = 0.7, high_performance = False, nreps = 100, 
                                   means = None, stds = None, DDM_id = 'angle',param_bounds = None):
    
    
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistribution: float
        Mean for the normal distribution to sample learning rates from.
    SD_LRdistribution: float
        Standard deviation for the normal distribution to sample learning rates from.
    mean_inverseTempdistribution: float
        Mean for the normal distribution to sample inverse temperatures from.
    SD_inverseTempdistribution: float
        Standard deviation for the normal distribution to sample inverse temperatures from.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the correlation value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if their correlation with the true parameters is minimum the cut_off.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    
    # RL design
    # start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1
    #divide process over multiple cores
    pool = Pool(processes = n_cpu)
    
    # Parameter distribution
    
# =============================================================================
#     # Incorrelation_repetition(means ,stds , 
#                                      param_bounds , 
#                                      npp = 150, 
#                                      ntrials = 450, DDM_id = "angle", rep=1, nreps = 250, ncpu = 6):
# =============================================================================
    out = pool.starmap(Functions_DDM.Incorrelation_repetition, [(means,stds, param_bounds, 
                                                   npp, ntrials,DDM_id,
                                                   rep, nreps, n_cpu) for rep in range(nreps)])
    pool.close()
    pool.join()

    allreps_output = pd.DataFrame(out, columns = ['correlations'])


    power_estimate = np.mean((allreps_output['correlations'] >= cut_off)*1)
    print(str("\nProbability to obtain a correlation(true_param, param_estim) >= {}".format(cut_off)
          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate*100)))

    return allreps_output, power_estimate

#%% 
import os, sys

if __name__ == '__main__':
# =============================================================================
#     criterion = sys.argv[1:]
#     assert len(criterion) == 1
# =============================================================================
    criterion = ['IC']
    criterion = criterion[0]
    DDM_id = 'angle'
    param_bounds = np.array(ssms.config.model_config['angle']['param_bounds'])
    

    InputFile_name = "InputFile_{}.csv".format(criterion)
    InputFile_path = os.path.join(os.getcwd(), InputFile_name)
    InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
    if InputParameters.shape[1] == 1: InputParameters = pd.read_csv(InputFile_path, delimiter = ';')	# depending on how you save the csv-file, the delimiter should be "," or ";". - This if-statement ensures that the correct delimiter is used. 
    InputDictionary = InputParameters.to_dict()
#%% start PA loop
    for row in range(InputParameters.shape[0]): 
        #Calculate how long it takes to do a power estimation
        start_time = datetime.now()
        print("Power estimation started at {}.".format(start_time))

        #Extract all values that are the same regardless of the criterion used
        ntrials = InputDictionary['ntrials'][row]
        nreps = InputDictionary['nreps'][row]                                                                                                                                        
        full_speed = InputDictionary['full_speed'][row]
        output_folder = InputDictionary['output_folder'][row]
# =============================================================================
#         # experiment setting, not for DDM
#         nreversals = InputDictionary['nreversals'][row]
#         reward_probability = InputDictionary['reward_probability'][row]
#         
# =============================================================================

         
# =============================================================================
#         # check parameters
#         variables_fine = check_input_parameters(ntrials, nreversals, reward_probability, full_speed, criterion, output_folder)
#         if variables_fine == 0: quit()
# =============================================================================
        
        #if not os.path.isdir(output_folder): 
        #    print('output_folder does not exist, please adapt the csv-file')
        #    quit()
#%% IC init
# switch critierion
        if criterion == "IC":
            npp = InputDictionary['npp'][row]
            tau = InputDictionary['tau'][row]
            
# =============================================================================
#             # distribution of parameters
#             meanLR, sdLR = InputDictionary['meanLR'][row], InputDictionary['sdLR'][row]
#             meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]

# =============================================================================
            means = []
            stds = []
            for DtbName, DtbValues in InputDictionary.items():
                if ("mean" in DtbName):
                    means.append(DtbValues[row])
                if ("std" in DtbName):
                    stds.append(DtbValues[row])
            means = np.array(means)    
            stds = np.array(stds) 
#%% power analysis (working)
            output, power_estimate = power_estimation_Incorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                                  cut_off = tau,high_performance = full_speed, 
                                                                  means = means, stds=stds,
                                                                  param_bounds = param_bounds,
                                                                  DDM_id = DDM_id)
            output.to_csv(os.path.join(output_folder, 'OutputIC{}T{}N{}M.csv'.format(ntrials,
                                                                                      npp, nreps)))
#%%                
            
            if HPC == False:
                fig, axes = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output["correlations"], label = "Correlations", ax = axes)
                fig.suptitle("Pr(Correlation >= {}) \nwith {} pp, {} trials)".format(tau, npp, ntrials), fontweight = 'bold')
                axes.set_title("Power = {}% \nbased on {} reps".format(np.round(power_estimate*100, 2), nreps))
                axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
#%%
        elif criterion == "GD":
            npp_pergroup = InputDictionary['npp_group'][row]
            npp = npp_pergroup*2
            meanLR_g1, sdLR_g1 = InputDictionary['meanLR_g1'][row], InputDictionary['sdLR_g1'][row]
            meanLR_g2, sdLR_g2 = InputDictionary['meanLR_g2'][row], InputDictionary['sdLR_g2'][row]
            meanInverseT_g1, sdInverseT_g1 = InputDictionary['meanInverseTemperature_g1'][row], InputDictionary['sdInverseTemperature_g1'][row]
            meanInverseT_g2, sdInverseT_g2 = InputDictionary['meanInverseTemperature_g2'][row], InputDictionary['sdInverseTemperature_g2'][row]
            typeIerror = InputDictionary['TypeIerror'][row]
            # Calculate tau based on the typeIerror and the df
            tau = -stat.t.ppf(typeIerror/2, npp-1)
            s_pooled = np.sqrt((sdLR_g1**2 + sdLR_g2**2) / 2)
            cohens_d = np.abs(meanLR_g1-meanLR_g2)/s_pooled

            output, power_estimate = power_estimation_groupdifference(npp_per_group = npp_pergroup, ntrials = ntrials,
                                               nreps = nreps, typeIerror = typeIerror, high_performance = full_speed,
                                               nreversals = nreversals, reward_probability = reward_probability,
                                               mean_LRdistributionG1 = meanLR_g1, SD_LRdistributionG1 = sdLR_g1,
                                               mean_LRdistributionG2 = meanLR_g2, SD_LRdistributionG2=sdLR_g2,
                                               mean_inverseTempdistributionG1 = meanInverseT_g1, SD_inverseTempdistributionG1 = sdInverseT_g1,
                                               mean_inverseTempdistributionG2 = meanInverseT_g2, SD_inverseTempdistributionG2 = sdInverseT_g2)
            output.to_csv(os.path.join(output_folder, 'OutputGD{}SD{}T{}R{}N{}M{}ES.csv'.format(np.round(s_pooled,2),
                                                                                                ntrials,
                                                                                      nreversals,
                                                                                      npp, nreps, np.round(cohens_d,2))))
            if HPC == False:
                fig, axes = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output["Statistic"], label = "T-statistic", ax = axes)
                fig.suptitle("Pr(T-statistic > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold')
                axes.set_title("Power = {}% \nbased on {} reps with Cohen's d = {}".format(np.round(power_estimate*100, 2), nreps, np.round(cohens_d,2)))
                axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')

        elif criterion == "EC":
            npp = InputDictionary['npp'][row]
            meanLR, sdLR = InputDictionary['meanLR'][row], InputDictionary['sdLR'][row]
            meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
            True_correlation = InputDictionary['True_correlation'][row]
            typeIerror = InputDictionary['TypeIerror'][row]
            s_pooled = sdLR

            beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
            tau = -beta_distribution.ppf(typeIerror/2)

            output, power_estimate = power_estimation_Excorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                                  typeIerror = typeIerror,
                                               high_performance = full_speed, nreversals = nreversals,
                                               reward_probability = reward_probability, mean_LRdistribution = meanLR,
                                               SD_LRdistribution = sdLR, mean_inverseTempdistribution = meanInverseT,
                                               SD_inverseTempdistribution = sdInverseT, True_correlation = True_correlation)
            output.to_csv(os.path.join(output_folder, 'OutputEC{}SD{}TC{}T{}R{}N{}M.csv'.format(s_pooled, True_correlation, ntrials,
                                                                                      nreversals,
                                                                                      npp, nreps)))
            if HPC == False:
                fig, axes = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output["Statistic"], label = "Correlation", ax = axes)
                fig.suptitle("Pr(Correlation > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp, ntrials), fontweight = 'bold')
                axes.set_title("Power = {}% \nbased on {} reps with true correlation {}".format(np.round(power_estimate*100, 2), nreps, True_correlation))
                axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')



        else: print("Criterion not found")
        #final adaptations to the output figure & store the figure
        if HPC == False:
            fig.legend(loc = 'center right')
            fig.tight_layout()
            fig.savefig(os.path.join(output_folder, 'Plot{}{}T{}R{}N{}M{}.jpg'.format(criterion,
                                                                                    np.round(s_pooled, 2),
                                                                                    ntrials, nreversals,
                                                                                    npp, nreps)))

        # measure how long the power estimation lasted
        end_time = datetime.now()
        print("\nPower analysis ended at {}; run lasted {} hours.".format(end_time, end_time-start_time))
