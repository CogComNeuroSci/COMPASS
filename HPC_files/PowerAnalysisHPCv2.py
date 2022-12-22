# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:04:23 2021

@author: maudb
"""
HPC = True

import numpy as np
import pandas as pd
from multiprocessing import Process, cpu_count, Pool, Queue
from FunctionsHPCv2 import create_design, Incorrelation_repetition, groupdifference_repetition, check_input_parameters, Excorrelation_repetition
from scipy import optimize
from scipy import stats as stat
from datetime import datetime
import os





def power_estimation_Incorrelation(npp = 30, ntrials = 480, nreversals = 12, cut_off = 0.7, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LRdistribution = 0.5, SD_LRdistribution = 0.1,
                                 mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0,
                                 data_dir = os.getcwd(), irep = 0):
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the proportion failed estimates on each repetition and the correlation value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the power to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if their correlation with the true parameters is minimum the cut_off.
    Power is calculated using a simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)

    # pool = Pool(processes = n_cpu)
    LR_distribution = np.array([mean_LRdistribution, SD_LRdistribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])

    Incorrelation_repetition(inverseTemp_distribution, LR_distribution, npp, ntrials,
                                                 start_design, data_dir, irep)

def power_estimation_groupdifference(npp_per_group = 20, ntrials = 480, nreps = 100, typeIerror = 0.05,
                                     high_performance = False, nreversals = 12, reward_probability = 0.8,
                                     mean_LRdistributionG1 = 0.5, SD_LRdistributionG1 = 0.1,
                                     mean_LRdistributionG2 = 0.5, SD_LRdistributionG2 = 0.1,
                                     mean_inverseTempdistributionG1 = 2.0, SD_inverseTempdistributionG1 = 1.0,
                                     mean_inverseTempdistributionG2 = 2.0, SD_inverseTempdistributionG2 = 1.0,
                                     data_dir = os.getcwd(), irep = 0):
    """
    Parameters
    ----------
    npp_per_group : integer
        Number of participants per group in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the proportion failed estimates on each repetition and the p-value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (significant group difference found) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the power to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if they correctly reveal the group difference when a true group difference of size 'cohens_d' exists.
    Power is calculated using a simulation-based approach.
    """

    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability, irep = irep, data_dir = data_dir)

    LR_distributions = np.array([[mean_LRdistributionG1, SD_LRdistributionG1], [mean_LRdistributionG2, SD_LRdistributionG2]])
    inverseTemp_distributions = np.array([[mean_inverseTempdistributionG1, SD_inverseTempdistributionG1],
                                          [mean_inverseTempdistributionG2, SD_inverseTempdistributionG2]])

    groupdifference_repetition(inverseTemp_distributions, LR_distributions, npp_per_group,
                                                 ntrials, start_design, data_dir, irep)

#%%
def power_estimation_Excorrelation(npp = 100, ntrials = 480, nreversals = 12, typeIerror = 0.05, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LRdistribution = 0.5, SD_LRdistribution = 0.1,
                                 mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0, True_correlation = .5,
                                 data_dir = os.getcwd(), irep = 0):
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
    Function that actually calculates the power to obtain significant correlations with external measures.
    Parameter estimates are considered to be adequate if correctly reveal a significant correlation when a significant correlation.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)

    #divide process over multiple cores
    LR_distribution = np.array([mean_LRdistribution, SD_LRdistribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])
    Excorrelation_repetition(inverseTemp_distribution, LR_distribution, True_correlation, npp, ntrials,
                                                 start_design, data_dir, irep)





import os, sys

input_parameters = sys.argv[1:]
assert len(input_parameters) == 2
criterion = input_parameters[0]
irep = input_parameters[1]

start_time = datetime.now()
print("Power estimation started at {}.".format(start_time))

InputFile_name = "InputFile_{}_simulations.csv".format(criterion)
InputFile_path = os.path.join(os.getcwd(), InputFile_name)
InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
InputDictionary = InputParameters.to_dict()
print(InputDictionary)

# durations = np.array([])
for row in range(InputParameters.shape[0]):
    #Calculate how long it takes to do a power estimation
    rep_starttime = datetime.now()

    #Extract all values that are the same regardless of the criterion used
    ntrials = InputDictionary['ntrials'][row]
    nreversals = InputDictionary['nreversals'][row]
    reward_probability = InputDictionary['reward_probability'][row]
    nreps = InputDictionary['nreps'][row]
    full_speed = InputDictionary['full_speed'][row]
    output_folder = InputDictionary['output_folder'][row]


    output_dir = os.path.join(output_folder, 'Output')

    if criterion == "IC":
        npp = InputDictionary['npp'][row]
        meanLR, sdLR = InputDictionary['meanLR'][row], InputDictionary['sdLR'][row]
        meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
        tau = InputDictionary['tau'][row]
        s_pooled = sdLR

        Results_folder = os.path.join(output_dir, "Results{}{}SD{}T{}R{}N".format(criterion, s_pooled,
                                                                                           ntrials, nreversals, npp))
        # if not os.path.isdir(Results_folder): os.makedirs(Results_folder)

        power_estimation_Incorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                              cut_off = tau,
                                           high_performance = full_speed, nreversals = nreversals,
                                           reward_probability = reward_probability, mean_LRdistribution = meanLR,
                                           SD_LRdistribution = sdLR, mean_inverseTempdistribution = meanInverseT,
                                           SD_inverseTempdistribution = sdInverseT, data_dir = Results_folder, irep = irep)





    elif criterion == "GD":
        npp_pergroup = InputDictionary['npp_group'][row]
        npp = npp_pergroup*2
        meanLR_g1, sdLR_g1 = InputDictionary['meanLR_g1'][row], InputDictionary['sdLR_g1'][row]
        meanLR_g2, sdLR_g2 = InputDictionary['meanLR_g2'][row], InputDictionary['sdLR_g2'][row]
        meanInverseT_g1, sdInverseT_g1 = InputDictionary['meanInverseTemperature_g1'][row], InputDictionary['sdInverseTemperature_g1'][row]
        meanInverseT_g2, sdInverseT_g2 = InputDictionary['meanInverseTemperature_g2'][row], InputDictionary['sdInverseTemperature_g2'][row]
        typeIerror = InputDictionary['TypeIerror'][row]
        # Calculate tau based on the typeIerror and the df
        tau = stat.t.ppf(1-typeIerror, npp_pergroup*2-1)
        s_pooled = np.sqrt((sdLR_g1**2 + sdLR_g2**2) / 2)
        cohens_d = np.abs(meanLR_g1-meanLR_g2)/s_pooled


        Results_folder = os.path.join(output_dir, "Results{}{}SD{}ES{}T{}R{}N".format(criterion,
                                                                                         np.round(s_pooled, 2),
                                                                                         np.round(cohens_d, 2),
                                                                                           ntrials, nreversals, npp))
        if not os.path.isdir(Results_folder): os.makedirs(Results_folder)

        power_estimation_groupdifference(npp_per_group = npp_pergroup, ntrials = ntrials,
                                           nreps = nreps, typeIerror = typeIerror, high_performance = full_speed,
                                           nreversals = nreversals, reward_probability = reward_probability,
                                           mean_LRdistributionG1 = meanLR_g1, SD_LRdistributionG1 = sdLR_g1,
                                           mean_LRdistributionG2 = meanLR_g2, SD_LRdistributionG2=sdLR_g2,
                                           mean_inverseTempdistributionG1 = meanInverseT_g1, SD_inverseTempdistributionG1 = sdInverseT_g1,
                                           mean_inverseTempdistributionG2 = meanInverseT_g2, SD_inverseTempdistributionG2 = sdInverseT_g2,
                                           data_dir = Results_folder, irep = irep)



    elif criterion == "EC":
        npp = InputDictionary['npp'][row]
        meanLR, sdLR = InputDictionary['meanLR'][row], InputDictionary['sdLR'][row]
        meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
        True_correlation = InputDictionary['True_correlation'][row]
        typeIerror = InputDictionary['TypeIerror'][row]
        s_pooled = sdLR

        beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
        tau = -beta_distribution.ppf(typeIerror)

        Results_folder = os.path.join(output_dir, "Results{}SD{}ES{}T{}R{}N{}".format(criterion,
                                                                                            np.round(s_pooled, 2),
                                                                                            np.round(True_correlation, 2),
                                                                                            ntrials, nreversals, npp))

        power_estimation_Excorrelation(npp = npp, ntrials = ntrials, nreps = nreps,typeIerror = typeIerror,
                                           high_performance = full_speed, nreversals = nreversals,
                                           reward_probability = reward_probability, mean_LRdistribution = meanLR,
                                           SD_LRdistribution = sdLR, mean_inverseTempdistribution = meanInverseT,
                                           SD_inverseTempdistribution = sdInverseT, True_correlation = True_correlation,
                                           data_dir = Results_folder, irep = irep)

    else: print("Criterion not found")

    rep_endtime = datetime.now()
    rep_duration = rep_endtime - rep_starttime
    print("Repetition with {} trials, {} pp lasted {}".format(ntrials, npp, rep_duration))
    # durations = np.append(durations, rep_duration)

# duration_file = os.path.join(output_dir, "Durations{}_rep{}.npy".format(criterion, irep))
# np.save(duration_file, durations)
# measure how long the power estimation lasted
end_time = datetime.now()
print("\nPower analysis ended at {}; run lasted {} hours.".format(end_time, end_time-start_time))
