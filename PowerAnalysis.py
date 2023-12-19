
"""
Created on Tue Dec 14 11:04:23 2021
MOdified on june 2023

@author: maudb, Luning He
"""
import os,sys
# os.chdir('D:/horiz/IMPORTANT/0study_graduate/Pro_COMPASS/COMPASS_DDM')

#%%
HPC = 0
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Functions_DDM import Incorrelation_repetition_DDM, Groupdifference_repetition_DDM, Excorrelation_repetition_DDM
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
def power_estimation_Incorrelation_DDM(means = None, stds=None, DDM_id = "ddm",
                                       npp = 30, ntrials = 30, nreps = 6,
                                       cut_off = 0.7, high_performance = 1,method = "Nelder-Mead"):
    
    
    """
    Parameters
    ----------
    means: array
        Means of distribution from which true parameters are sampled
    stds: array
        Stds of distribution from which true parameters are sampled
    DDM_id: string
        Index of DDM model which should be matched with ssms package
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
    method: string
        Method of optimization, including: "Nelder-Mead" and "Brute"

    Returns
    -------
    allreps_output : DataFrame
        Pandas dataframe containing statistics from each nrep of all parameters
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

    print("Optimization Method:",method)

    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])

    out_AllReturns = pool.starmap(Incorrelation_repetition_DDM, [(means,stds, param_bounds, 
                                                   npp, ntrials,DDM_id,method,
                                                   rep, nreps, n_cpu) for rep in range(nreps)])


    
    # out = Incorrelation_repetition(means,stds, param_bounds, 
    #                                npp, ntrials,DDM_id,method,
    #                               1, nreps, n_cpu) 
    
    pool.close()
    pool.join()
    # allreps_output = pd.DataFrame(out_AllReturns, columns = ['Statistic','True_Par','Esti_Par', 'ACC_out', 'RT_out'])

    out = [IC_Sta[0] for IC_Sta in out_AllReturns]
    True_out = [IC_Sta[1] for IC_Sta in out_AllReturns]
    cn = ssms.config.model_config[DDM_id]['params']+["ACC","RT"]
    allreps_output = pd.DataFrame(np.array(out).reshape(nreps,len(means)+2),columns=cn)
    power_estimate = pd.DataFrame(np.empty((1,len(means))),columns=ssms.config.model_config[DDM_id]['params'])
   
    for p in range(len(means)):
        power_estimate.iloc[0,p] = np.mean((allreps_output.iloc[:,p] >= cut_off)*1)
#    print(str("\nProbability to obtain a correlation(true_param, param_estim) >= {}".format(cut_off)
#          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate*100)))

    return allreps_output, power_estimate
def power_estimation_Excorrelation_DDM(means,stds,par_ind,DDM_id,true_correlation = 0.5,
                                       npp = 100, ntrials = 480, nreps = 100,
                                       typeIerror = 0.05, high_performance = True, ncpu = 6):
    """

    Parameters
    ----------
    means: array
        Means of distribution from which true parameters are sampled
    stds: array
        Stds of distribution from which true parameters are sampled
    par_ind: integer
        Parameter index of the parameter of interest, according to the order from ssms.
        START FROM 0. E.g., par_ind = 0, corresponding to "v"
    DDM_id: 
        Index of DDM model which should be matched with ssms package
    True_correlation: float
        The hypothesized correlation between the learning rate and the external measure theta.    
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    typeIerror : float
        Critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    ncpu: integer
        number of cpu used


    Returns
    -------
    allreps_output : datafram
        Pandas dataframe containing estimated and statistics of the parameter of interest: estimated correlation coefficient, estimated p-value, true correlation coefficient, true p-value
    power_estimate: dataframe
        Results of power analysis, including: the name of parameter calulated, correlation cut-off value(i.e., tau), trure p-value, conventional power.
    Description
    -----------
    Function that actually calculates the probability to obtain significant correlations with external measures.
    Parameter estimates are considered to be adequate if correctly reveal a significant correlation when a significant correlation.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    #Use beta_distribution to determine the p-value for the hypothesized correlation
    beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
    true_pValue = 1-beta_distribution.cdf(True_correlation)
    tau = -beta_distribution.ppf(typeIerror/2)


    # Specifically, beta.pdf(x, a, b, loc, scale) is identically equivalent to beta.pdf(y, a, b) / scale with y = (x - loc) / scale. 
    # Note that shifting the location of a distribution does not make it a “noncentral” distribution; 
    # noncentral generalizations of some distributions are available in separate classes.
    
    #compute conventional power
    noncentral_beta = stat.beta((npp/2)-1, (npp/2)-1, loc = -1+True_correlation, scale = 2)
    conventional_power = 1-noncentral_beta.cdf(tau)

    print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(conventional_power*100,2))))
    print(str("\nThe correlation cut-off value is: {}".format(np.round(tau,2))))
    print(str("\np-value for true correlation is :{}".format(np.round(true_pValue,5))))

    #divide process over multiple cores
    pool = Pool(processes = n_cpu)

    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    out = pool.starmap(Excorrelation_repetition_DDM, [(means,stds,
                                                    param_bounds,par_ind, DDM_id,true_correlation,
                                                    npp, ntrials, rep, nreps, ncpu) for rep in range(nreps)])


    pool.close()
    pool.join()

    allreps_output = pd.DataFrame(out, columns = ['Esti_r', 'Esti_pValue', 'True_r', 'True_pValue',"ACC", "RT"])

    #Compute power if estimates would be perfect.

    #Compute power for correlation with estimated parameter values.
    
    power_estimate =  pd.DataFrame(np.empty((1,4)), columns=["power_"+ssms.config.model_config[DDM_id]['params'][par_ind],"tau",'true_pValue','conventional_power'])
    power_estimate.iloc[0,0] = np.mean((allreps_output['Esti_pValue'] <= typeIerror/2)*1)
    power_estimate.iloc[0]['tau'] = tau
    power_estimate.iloc[0]['true_pValue'] = true_pValue
    power_estimate.iloc[0]['conventional_power'] = conventional_power
    
    print(str("\nProbability to obtain a significant correlation between model parameter {} and an external measure that is {} correlated".format(ssms.config.model_config[DDM_id]['params'][par_ind],True_correlation)
        + " with {} trials and {} participants: {}%".format(ntrials, npp, np.round(power_estimate.iloc[0,0]*100,2))))


    return allreps_output, power_estimate
def power_estimation_groupdifference_DDM(cohens_d, means_g1,means_g2,stds_g1,stds_g2,DDM_id, par_ind,
                                        npp_per_group = 40, ntrials = 100,
                                        nreps = 250, typeIerror = 0.05, high_performance = 1):

    """
    Parameters
    ----------
    means_g1: array
        means of the distribution from which samples true parameters of group 1
    means_g2: array
        means of the distribution from which samples true parameters of group 2
    stds_g1: array
        stds of the distribution from which samples true parameters of group 1
    stds_g2: array
        stds of the distribution from which samples true parameters of group 2
    DDM_id: string
        Index of DDM model which should be matched with ssms package
    par_ind:
        Parameter index of the parameter of interest, according to the order from ssms.
        START FROM 0. E.g., par_ind = 0, corresponding to "v"
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
    
    Returns
    -------
    allreps_output : dataframe
        Pandas dataframe containing the t-value and p-value based on t-test of estimated parameters on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which a significant group difference was found divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if they correctly reveal the group difference when a true group difference of size 'cohens_d' exists.
    Power is calculated using a Monte Carlo simulation-based approach.
    """

    if high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    if __name__ == '__main__':
        
        #Use t_distribution to determine the p-value for the hypothesized cohen's d
        true_pValue = 1-stat.t.cdf(cohens_d*np.sqrt(npp_per_group), (npp_per_group-1)*2)
        tau = -stat.t.ppf(typeIerror/2, (npp_per_group-1)*2)
        
        #Compute conventional power
        conventional_power = 1-stat.nct.cdf(tau, (npp_per_group-1)*2, cohens_d*np.sqrt(npp_per_group))

        print("\nProbability to obtain a significant group difference under conventional power implementation: {}%".format(np.round(conventional_power*100,2)))
        print(str("\nEstimated effect size d = : {}".format(np.round(cohens_d,3))))
        print(str("\nThe t-distribution cut-off value is: {}".format(np.round(tau,3))))
        print(str("\np-value for given cohen's d is :{}".format(np.round(true_pValue,5))))
        #divide process over multiple cores

        pool = Pool(processes = n_cpu)
        param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
        out = pool.starmap(Groupdifference_repetition_DDM, [(means_g1, stds_g1,means_g2, stds_g2,DDM_id, par_ind,param_bounds,
                                   npp_per_group, ntrials, rep, nreps, n_cpu , False) for rep in range(nreps)])
        # before calling pool.join(), should call pool.close() to indicate that there will be no new processing
        pool.close()
        pool.join()

        allreps_output = pd.DataFrame(out, columns = ['Statistic', 'PValue','ACC_g1','ACC_g2','RT_g1','RT_g2'])

        # check for which % of repetitions the group difference was significant
        # note that we're working with a one-sided t-test (if interested in two-sided need to divide the p-value obtained at each rep with 2)
        power_estimate =  pd.DataFrame(np.empty((1,4)), columns=["power_"+ssms.config.model_config[DDM_id]['params'][par_ind],"tau",'true_pValue','conventional_power'])
        power_estimate.iloc[0,0] = np.mean((allreps_output['PValue'] <= typeIerror/2))
        power_estimate.iloc[0]['tau'] = tau
        power_estimate.iloc[0]['true_pValue'] = true_pValue
        power_estimate.iloc[0]['conventional_power'] = conventional_power
        print(str("\nProbability to detect a significant group difference when the estimated effect size d = {}".format(np.round(cohens_d,3))
              + " with {} trials and {} participants per group: {}%".format(ntrials,
                                                                         npp_per_group, np.round(power_estimate.iloc[0,0]*100,2))))
        return allreps_output, power_estimate

#%% 
def GetMeansStd(InputDictionary):
    """
    Parameters
    ----------
    InputDictionary： Dictionary
        Configures from the input csv file

    Returns
    -------
    means : array
        Each element is an extracted mean of a parameter in input filem, with the same order as defined in the csv
    stds: array
        Each element is an extracted std of a parameter in input filem, with the same order as defined in the csv

    Description
    -----------
    Function that extract means and stds from the input file based on cols with "mean_" and "std_".
    """
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

    criterion = sys.argv[1:]
    if not criterion:
        criterion = ["IC_DDM"]
    assert len(criterion) == 1
    criterion = criterion[0]
    # criterion = criterion[0]
    
   
    InputFile_name = "InputFile_{}.csv".format(criterion)
    InputFile_path = os.path.join(os.getcwd(), InputFile_name)
    InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
    if InputParameters.shape[1] == 1: InputParameters = pd.read_csv(InputFile_path, delimiter = ';')	# depending on how you save the csv-file, the delimiter should be "," or ";". - This if-statement ensures that the correct delimiter is used. 
    
    InputDictionary = InputParameters.to_dict()
    range_npp = []
    range_ntrials = []
    for row in range(InputParameters.shape[0]): 
        
        #Calculate how long it takes to do a power estimation
        start_time = datetime.now()
        print("Power estimation started at {}.".format(start_time))

        #Extract all values that are the same regardless of the criterion used
        
        ntrials =  int(InputDictionary['ntrials'][row])
        nreps = int(InputDictionary['nreps'][row])                                                                                                                               
        full_speed = InputDictionary['full_speed'][row]
        output_folder = InputDictionary['output_folder'][row]
     
# =============================================================================
#         # check parameters
#         variables_fine = check_input_parameters(ntrials, nreversals, reward_probability, full_speed, criterion, output_folder)
#         if variables_fine == 0: quit()
# =============================================================================
        
        if not os.path.isdir(output_folder): 
            print('output_folder does not exist, please adapt the csv-file')
            # quit()
            sys.exit(0)
# IC 
        if criterion == "IC_DDM":
            npp = int(InputDictionary['npp'][row])
            tau = InputDictionary['tau'][row]
            DDM_id = InputDictionary['model'][row]
            means,stds = GetMeansStd(InputDictionary)


            print("\nStart IC analysis for DDM model\n")
            print("model: {}".format(DDM_id))
            print("trials: {}".format(ntrials))
            print("participants: {}".format(npp))
            print("tau: {}".format(tau))
            output, power_estimate = power_estimation_Incorrelation_DDM(means = means, stds=stds, DDM_id = DDM_id,
                                                                        npp = npp, ntrials = ntrials, nreps = nreps,
                                                                        cut_off = tau, high_performance = full_speed)
            
            # output = pd.concat([output,InputParameters])
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            InputParameters.to_csv(os.path.join(output_folder, 'InputIC{}T{}N{}M_{}.csv'.format(ntrials,npp, nreps, time)))
            output.to_csv(os.path.join(output_folder, 'OutputIC{}T{}N{}M_{}.csv'.format(ntrials,npp, nreps,time)))
            # power_estimate = pd.concat([power_estimate,InputParameters])
            power_estimate.to_csv(os.path.join(output_folder, 'PowerIC{}T{}N{}M_{}.csv'.format(ntrials,npp, nreps,time)))
            if HPC == False:
                Parameters = ssms.config.model_config[DDM_id]["params"]
                for p in Parameters :
                    plt.tight_layout() 
                    fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize=(8, 8))
                    # sns.set_theme(style = "white",font_scale=1.4)
                    sns.kdeplot(output[p].dropna(axis = 0), ax = axes)
                    fig.suptitle("Pr(Correlation >= {}) with {} pp, {} trials)".format(tau, npp, ntrials), fontweight = 'bold',fontsize = 25)

                    power_value = power_estimate[p].dropna(axis = 0).values[0]
                    axes.set_title("Power = {}% based on {} reps".format(np.round(power_value*100, 2), nreps),fontsize = 25)
                    axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                    axes.tick_params(labelsize=20)
                    axes.set_xlabel('Correlations of '+p,fontsize = 20)
                    axes.set_ylabel('Density',fontsize = 20)

                    file_name = 'PowerIC{}T{}N{}M_{}_{}.png'.format(ntrials,npp,nreps,p,time)
                    plt.savefig(output_folder+file_name,bbox_inches='tight')
                    plt.show(block=False) 
# EC
        elif criterion == "EC_DDM":
            npp = InputDictionary['npp'][row]
            means,stds = GetMeansStd(InputDictionary)
            True_correlation = InputDictionary['True_correlation'][row]
            typeIerror = InputDictionary['TypeIerror'][row]
            par_ind = InputDictionary['par_ind'][row]

            s_pooled = stds[par_ind]

            beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
            tau = -beta_distribution.ppf(typeIerror/2)

            DDM_id = InputDictionary['model'][row]
            param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])

            print("\nStart EC analysis for DDM model\n")
            print("model: {}".format(DDM_id))
            print("trials: {}".format(ntrials))
            print("participants: {}".format(npp))
            print("parameter index: {}".format(par_ind))
            print("True_correlation: {}".format(True_correlation))
            output, power_estimate = power_estimation_Excorrelation_DDM(means,stds,par_ind,DDM_id,
                                                                        True_correlation,npp, ntrials, nreps,
                                                                        typeIerror, high_performance = True, ncpu = 6)
            
            # output = pd.concat([output,InputParameters])
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            InputParameters.to_csv(os.path.join(output_folder, 'InputEC{}P{}SD{}TC{}T{}N{}M_{}.csv'.format(par_ind,s_pooled, True_correlation, ntrials,
                                                                                      npp, nreps,time)))           
            output.to_csv(os.path.join(output_folder, 'OutputEC{}P{}SD{}TC{}T{}N{}M_{}.csv'.format(par_ind,s_pooled, True_correlation, ntrials,
                                                                                      npp, nreps,time)))
            
            # power_estimate = pd.concat([pd.DataFrame(power_estimate),InputParameters])
            power_estimate.to_csv(os.path.join(output_folder, 'PowerEC{}P{}SD{}TC{}T{}N{}M_{}.csv'.format(par_ind,s_pooled, True_correlation, ntrials,
                                                                                      npp, nreps,time)))     
            
            if HPC == False:

                # sns.kdeplot(output["Statistic"], label = "Correlation", ax = axes)
                # fig.suptitle("Pr(Correlation > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp, ntrials), fontweight = 'bold')
                # axes.set_title("Power = {}% \nbased on {} reps with true correlation {}".format(np.round(power_estimate*100, 2), nreps, True_correlation))
                # axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
            
                plt.tight_layout() 
                p_n = "power_"+ssms.config.model_config[DDM_id]['params'][par_ind]
                fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize=(10,10))
                # sns.set_theme(style = "white",font_scale=1.4)
                sns.kdeplot(output["Esti_r"].dropna(axis = 0),label = "Correlation", ax = axes)
                fig.suptitle("Pr(Correlation > {}) considering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp, ntrials), fontweight = 'bold',fontsize = 25)
                power_value = power_estimate[p_n].dropna(axis = 0).values[0]

                axes.set_title("Power = {} % based on {} reps with true correlation {}".format(np.round(power_value*100, 2), nreps, True_correlation),fontsize = 25)
                axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                axes.tick_params(labelsize=20)
                axes.set_xlabel('Correlations of '+ p_n, fontsize = 20)
                axes.set_ylabel('Density',fontsize = 20)
                file_name = 'PowerEC{}P{}SD{}TC{}T{}N{}M_{}.png'.format(par_ind,s_pooled, True_correlation, ntrials,
                                                                                      npp, nreps,time)
                plt.savefig(output_folder+file_name,bbox_inches='tight') 
                plt.show(block=False) 

 
# DG
        elif criterion == "GD_DDM":
            npp_pergroup = InputDictionary['npp_group'][row]
            npp = npp_pergroup*2
            par_ind = InputDictionary['par_ind'][row]
            typeIerror = InputDictionary['TypeIerror'][row]
            means,stds = GetMeansStd(InputDictionary)

            DDM_id = InputDictionary['model'][row]
            param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])

            def GetGroupDistribution(raw_DistributionPar,par_ind,group):
                NotCompare_ind = np.ones(raw_DistributionPar.shape, dtype=bool)
                NotCompare_ind[par_ind] = False
                try :
                    Compare_ms_g = float(raw_DistributionPar[par_ind].split(",")[group-1])
                except:
                    print("Check the distribution and index of parameter you want to compare")
                    sys.exit(0)
                else:

                    ms_g = raw_DistributionPar[NotCompare_ind].astype(float)
                    ms_g = np.insert(ms_g, par_ind, Compare_ms_g)
                    return ms_g

            means_g1 = GetGroupDistribution(means,par_ind,group = 1)
            means_g2 = GetGroupDistribution(means,par_ind,group = 2)
            stds_g1 = GetGroupDistribution(stds,par_ind,group = 1)
            stds_g2 = GetGroupDistribution(stds,par_ind,group = 2)



            # Calculate tau based on the typeIerror and the df
            tau = -stat.t.ppf(typeIerror/2, npp-1)
            s_pooled = np.sqrt((stds_g1[par_ind]**2 + stds_g2[par_ind]**2) / 2)
            cohens_d = np.abs(means_g1[par_ind]-means_g2[par_ind])/s_pooled


            print("\nStart GD analysis for DDM model\n")
            print("model: {}".format(DDM_id))
            print("trials: {}".format(ntrials))
            print("participants: {}".format(npp))
            print("parameter index: {}".format(par_ind))
            print("Means of group 1: {}, stds of group 1: {}".format(means_g1,stds_g1))
            print("Means of group 2: {}, stds of group 2: {}".format(means_g2,stds_g2))

            output, power_estimate = power_estimation_groupdifference_DDM(cohens_d,means_g1,means_g2,stds_g1,stds_g2,DDM_id, par_ind,
                                                                      npp_per_group = npp_pergroup, ntrials = ntrials,
                                                                      nreps = nreps, typeIerror = typeIerror, high_performance = full_speed)


            # output = pd.concat([output,InputParameters])
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            InputParameters.to_csv(os.path.join(output_folder, 'InputGD{}P{}SD{}T{}N{}M{}ES_{}.csv'.format(par_ind,np.round(s_pooled,2),
                                                                                                ntrials,npp_pergroup, nreps, np.round(cohens_d,2),time)))
            output.to_csv(os.path.join(output_folder, 'OutputGD{}P{}SD{}T{}N{}M{}ES_{}.csv'.format(par_ind,np.round(s_pooled,2),
                                                                                                ntrials,npp_pergroup, nreps, np.round(cohens_d,2),time)))
            # power_estimate = pd.concat([pd.DataFrame(power_estimate),InputParameters])
            power_estimate.to_csv(os.path.join(output_folder, 'PowerGD{}P{}SD{}T{}N{}M{}_{}ES.csv'.format(par_ind,np.round(s_pooled,2),ntrials,
                                                                                      npp_pergroup, nreps,np.round(cohens_d,2),time)))
            if HPC == False:

                # fig, axes = plt.subplots(nrows = 1, ncols = 1)
                # sns.kdeplot(output["Statistic"], label = "T-statistic", ax = axes)
                # fig.suptitle("Pr(T-statistic > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold')
                # axes.set_title("Power = {}% \nbased on {} reps with Cohen's d = {}".format(np.round(power_estimate*100, 2), nreps, np.round(cohens_d,2)))
                # axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')

                # plt.tight_layout() 
                p_n = "power_"+ssms.config.model_config[DDM_id]['params'][par_ind]
                fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize=(10, 10))
                # sns.set_theme(style = "white",font_scale=1.4)
                sns.kdeplot(output["Statistic"].dropna(axis = 0),label = "Correlation", ax = axes)
                fig.suptitle("Pr(T-statistic > {}) considering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold',fontsize = 25)
                power_value = power_estimate[p_n].dropna(axis = 0).values[0]
                axes.set_title("Power = {} % based on {} reps with Cohen's d = {}".format(np.round(power_value*100, 2), nreps, np.round(cohens_d,2)),fontsize = 25)
                axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                axes.tick_params(labelsize=20)
                axes.set_xlabel('T-statistics of '+ p_n, fontsize = 20)
                axes.set_ylabel('Density',fontsize = 20)

                file_name = 'PowerGD{}P{}SD{}T{}N{}M{}ES_{}.png'.format(par_ind,np.round(s_pooled,2),ntrials,
                                                                                      npp_pergroup, nreps,np.round(cohens_d,2),time)
                
                plt.savefig(output_folder+file_name,bbox_inches='tight')  
                # plt.show() 

        else: print("Criterion not found")
        # # measure how long the power estimation lasted
        end_time = datetime.now()
        print("\nPower analysis ended at {}; run lasted {} hours.".format(end_time, end_time-start_time))
    
