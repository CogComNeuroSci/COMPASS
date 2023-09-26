
"""
Created on Tue Dec 14 11:04:23 2021
MOdified on june 2023

@author: maudb, Luning He
"""
import os,sys
# os.chdir('D:/horiz/IMPORTANT/0study_graduate/Pro_COMPASS/COMPASS_DDM')

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


#%% DDM functions
def power_estimation_Incorrelation(npp = 30, ntrials = 480, cut_off = 0.7, high_performance = False, nreps = 100, 
                                   means = None, stds = None, DDM_id = 'angle',param_bounds = None,method = "Nelder-Mead"):
    
    
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    means: 
    stds:
    DDM_id: 
    param_bounds:
    method:


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
    print("Start IC analysis")
    print("Optimization Method:",method)

    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    

    out_AllReturns = pool.starmap(Incorrelation_repetition, [(means,stds, param_bounds, 
                                                   npp, ntrials,DDM_id,method,
                                                   rep, nreps, n_cpu) for rep in range(nreps)])


    
    # out = Incorrelation_repetition(means,stds, param_bounds, 
    #                                npp, ntrials,DDM_id,method,
    #                               1, nreps, n_cpu) 
    
    pool.close()
    pool.join()

    out = [IC_Sta[0] for IC_Sta in out_AllReturns]

    allreps_output = pd.DataFrame(np.array(out).reshape(nreps,len(means)),columns=ssms.config.model_config[DDM_id]['params'])
    power_estimate = pd.DataFrame(np.empty((1,len(means))),columns=ssms.config.model_config[DDM_id]['params'])
    for p in range(len(means)):
        power_estimate.iloc[0,p] = np.mean((allreps_output.iloc[:,p] >= cut_off)*1)
#    print(str("\nProbability to obtain a correlation(true_param, param_estim) >= {}".format(cut_off)
#          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate*100)))

    return allreps_output, power_estimate

#%% 
def GetMeansStd(InputDictionary):
    means = []
    stds = []
# =============================================================================
#             # NOTE: the order of parameters should be corresponding to that in ssms
#             to see parameters included and its order: ssms.config.model_config[DDM_id]['params']    
# =============================================================================
    for DtbName, DtbValues in InputDictionary.items():
        if ("mean" in DtbName):
            means.append(DtbValues[row])
        if ("std" in DtbName):
            stds.append(DtbValues[row])
    means = np.array(means)    
    stds = np.array(stds) 
    return means,stds


if __name__ == '__main__':
# =============================================================================
#     criterion = sys.argv[1:]
#     assert len(criterion) == 1
# =============================================================================
    criterion = ['IC']
    criterion = criterion[0]
    modelclass = "DDM"
   
    InputFile_name = "InputFile_{}.csv".format(criterion)
    InputFile_path = os.path.join(os.getcwd(), InputFile_name)
    InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
    if InputParameters.shape[1] == 1: InputParameters = pd.read_csv(InputFile_path, delimiter = ';')	# depending on how you save the csv-file, the delimiter should be "," or ";". - This if-statement ensures that the correct delimiter is used. 
    
    InputDictionary = InputParameters.to_dict()

    for row in range(InputParameters.shape[0]): 
        #Calculate how long it takes to do a power estimation
        start_time = datetime.now()
        print("Power estimation started at {}.".format(start_time))

        #Extract all values that are the same regardless of the criterion used
        ntrials = InputDictionary['ntrials'][row]
        nreps = int(InputDictionary['nreps'][row])                                                                                                                                        
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
        
        if not os.path.isdir(output_folder): 
            print('output_folder does not exist, please adapt the csv-file')
            # quit()
            sys.exit(0)
#%% IC init
# switch critierion
        if criterion == "IC":
            npp = int(InputDictionary['npp'][row])
            tau = InputDictionary['tau'][row]
# =============================================================================
#             # define DDM models in ssms 
#             to see included models: list(ssms.config.model_config.keys())[:10]
# =============================================================================    
            if modelclass == "DDM":
                DDM_id = InputDictionary['model'][row]
                if DDM_id:
                    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
# =============================================================================
#             # distribution of parameters
#             meanLR, sdLR = InputDictionary['meanLR'][row], InputDictionary['sdLR'][row]
#             meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
# =============================================================================

            
            means,stds = GetMeansStd(InputDictionary)
    #%% power analysis (working)
            output, power_estimate = power_estimation_Incorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                                  cut_off = tau,high_performance = full_speed, 
                                                                  means = means, stds=stds,
                                                                  param_bounds = param_bounds,
                                                                  DDM_id = DDM_id)
            output = pd.concat([output,InputParameters])
            output.to_csv(os.path.join(output_folder, 'OutputIC{}T{}N{}M.csv'.format(ntrials,
                                                                                      npp, nreps)))
            power_estimate = pd.concat([power_estimate,InputParameters])
            power_estimate.to_csv(os.path.join(output_folder, 'PowerIC{}T{}N{}M.csv'.format(ntrials,
                                                                                      npp, nreps)))
#%%               
            
            if HPC == False:
                print('plot codes moved to plot.file')


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
            print('plot codes removed')
        #     fig.legend(loc = 'center right')
        #     fig.tight_layout()
        #     fig.savefig(os.path.join(output_folder, 'Plot{}{}T{}R{}N{}M{}.jpg'.format(criterion,
        #                                                                             np.round(s_pooled, 2),
        #                                                                             ntrials, nreversals,
        #                                                                             npp, nreps)))

        # # measure how long the power estimation lasted
        end_time = datetime.now()
        print("\nPower analysis ended at {}; run lasted {} hours.".format(end_time, end_time-start_time))





