
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:09:53 2021
@author: Maud
"""
import ssms
import numpy as np
import pandas as pd
import os, time
import math
from scipy import optimize
from scipy import stats as stat
import matplotlib.pyplot as plt
from Likelihoods import neg_likelihood
from ParameterEstimation import MLE


#This is to avoid warnings being printed to the terminal window
import warnings
warnings.filterwarnings('ignore')


#%% DDM functions IC
# functions for DDM
def Incorrelation_repetition(means,stds , 
                             param_bounds, 
                             npp = 150,ntrials = 450, 
                             DDM_id = "ddm", method = "Brute",rep=1, nreps = 250, ncpu = 6):
    """

    Parameters
    ----------

    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu:
        The amount of cpu available.

    Returns
    -------
    Statistic : float
        The correlation found between the true and recovered parameters this repetition.

    Description
    -----------
    Function to execute the parameter recovery analysis (Internal correlation criterion) once.
    This criterion prescribes that resources are sufficient when: correlation(true learning rates, recovered learning rates) >= certain cut-off.
    Thus, the statistic of interest is: correlation(true learning rates, recovered learning rates). This statistic is returned for execution of this function (thus for each repetition).
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:
        1. Create 'npp' hypothetical participants by defining 'npp' parameter sets.
            A parameter set consists of a value for the learning rate and a value for the inverse temperature.
            One population is assumed with the following parameter distributions:
                - learning rates ~ N(mean, sd)
                    --> mean = LR_distribution[0], sd = LR_distribution[1]
                - inverse temperatures ~ N(mean, sd)
                    --> mean = Temperature_distribution[0], sd = Temperature_distribution[1]
        2. Simulate data for each hypothetical participant (thus each parameter set)
            This is done by simulating responses using the Rescorla-Wagner model (RW-model) with the values of the free parameters = the parameter set of this hypothetical participant.
            This basic RW-model incorporates a delta-learning rule and a softmax choice rule.
            (for details on the basic RW-model see ... (github-link naar ReadME))
        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'recovered parameter values'
            This is done using the Maximum log-Likelihood estimation process in combination with the basic RW-model: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
        4. Calculate the Statistic of interest for this repetition of the parameter recovery analysis.
            The statistic that is calculated here is correlation(true learning rates, recovered learning rates).
    If this function is repeated a number of times and the value of the Statistic is stored each time, we can evaluate later on the power or probability to meet the proposed parameter recovery criterion (the internal correlation criterion) in a single study.
    """
    print("Sample:",rep,"/",nreps)
    if rep == 0:
        t0 = time.time()

    ####PART 1: parameter generation for all participants####
    # Define the True params that will be used for each pp in this rep

    # True_Par =  generate_parameters(means = means, stds = stds, 
    #                                 param_bounds = param_bounds, npp = npp)
    # True_Par.columns = ssms.config.model_config[DDM_id]['params']

    True_Par = pd.DataFrame(np.empty((npp,len(means))))
    True_Par.columns = ssms.config.model_config[DDM_id]['params']

    Esti_Par = pd.DataFrame(np.empty((npp,len(means))))
    Esti_Par.columns = ssms.config.model_config[DDM_id]['params']

    ACC_out = np.empty((npp,1))
    RT_out = np.empty((npp,1))

    waste_counter = 0

   ###########################################
   #          RESCALE THE RANGE OF A         #
   ###########################################
    param_bounds_Opti = param_bounds
    param_bounds_Opti[:,1] = param_bounds[:,1]*2 


    for pp in range(npp): # loop for participants
        ACC = 0
        ####Part 2: Data simulation for this participant####
        while ACC <= 0.50 or ACC >= 0.95 or RT >= 10:
        # generate the responses for this participant

            One_True_Par = generate_parameters(means = means, stds = stds, 
                                     param_bounds = param_bounds, npp = 1)



            responses = simulate_responses(np.array(One_True_Par),DDM_id,ntrials)
            responses = np.array(responses['rts'] * responses['choices'])
            # validation of parameters
            
            ACC = np.mean( responses*One_True_Par.iloc[0,0] > 0)            
            RT = np.mean(np.abs(responses))
            
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10: 
                waste_counter = waste_counter+1
         

        True_Par.iloc[pp,:] = np.array(One_True_Par)
        ACC_out[pp] = ACC
        RT_out[pp] = RT

        ####Part 3: parameter estimation for this participant####
        fun = neg_likelihood
        arg = (responses,DDM_id)
         # method = "Nelder-Mead"  or method=="Brute"
        # re-scaling parameter a
        Esti_Par.iloc[pp,:] = MLE(fun,arg,param_bounds_Opti,method,show = 0)     

        # print(Esti_Par[pp],neg_likelihood(Esti_Par[pp],arg)) 
        # print(np.array(True_Par.loc[pp]),neg_likelihood(True_Par.loc[pp],arg))


    Esti_Par['a'] = Esti_Par['a']/2
    ####Part 4: correlation between true & estimated learning rates####
    # if the estimation failed for a certain participant, delete this participant from the correlation estimation for this repetition
    Statistic = np.empty((1,len(means)))
    for p in range(len(means)):
        Statistic[0,p] = np.round(np.corrcoef(True_Par.iloc[:,p], Esti_Par.iloc[:,p])[0,1], 3)
        print("Statistic:",Statistic[0,p])

 
    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    
    return Statistic,True_Par,Esti_Par, ACC_out, RT_out


def generate_parameters(means,stds , 
                        param_bounds , 
                        npp = 150, multivariate = False, corr = False):
    """
    Parameters
    ----------
    means : 1 * n array 
        means of parameters in an array. n: number of parameters
    stds : 1 * n array 
        The standard deviation of the normal distribution from which parameters are drawn. The default is 0.1.
    param_bounds: 2 * n array
        min and max of parameters          
    multivariate: boolean, optional
        Put to true for the external correlation criterion such that values are drawn from multivariate normal distribution. The default is False.
    corr: boolean or float, optional
        The correlation for the external correlation criterion. For other criterions this is ignored. The default is False.
    Returns
    -------
    parameters : npp * len(means) numpy array 
        Array with shape ('size',) containing the parameters drawn from the normal distribution.

    Description
    -----------
    Function to draw 'npp' parameters from a normal distribution with mean 'mean' and standard deviation 'std'.
    Function is used to generate learning rate and inverse temperature parameters for each participant.
    No parameters get a value lower than or equal to 0.
    When the criterion is external correlation, learning rate and the external measure are drawn from a multivariate normal distribution.
    Here, the correlation is specified in the covariance matrix."""

    if multivariate:
        # draw 'npp' values from multivariate normal distribution with mean 'mean', standard deviation 'std' and correlation 'cor'
        #parameters =np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), npp),3)
        # while-loop: ensure no learning rate parameters get a value smaller than or equal to 0
       #  while np.any(parameters[:,0] <= 0):
            # to_replace = np.where(parameters[:,0] <= 0)[0]
            # parameters[to_replace, :] = np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), len(to_replace)),3)
            pass
    else:
        # sample all parameters 
        z = np.zeros((npp, len(means)))
        parameters = pd.DataFrame(z, 
                            dtype = np.float32)
        if len(means) != param_bounds.shape[1]:
            print('the name of model and parameters do not match')
            sys.exit(0)
            
        for p in range(len(means)):
            # draw 'npp' values from normal distribution with mean 'mean' and standard deviation 'std'
            parameters[p] = np.round(np.random.normal(loc = means[p], scale = stds[p], size = npp), 3)
            
            while max(parameters[p])>param_bounds[1][p] or min(parameters[p])<param_bounds[0][p]:
                outerID = np.logical_or(parameters[p] <= param_bounds[0][p],parameters[p] >= param_bounds[1][p])
                parameters[p][outerID] = np.round(np.random.normal(loc = means[p], scale = stds[p], size = sum(outerID)), 3)
                
            # while-loop: ensure no parameters get a value smaller than or equal to 0
# =============================================================================
#             while (np.any(np.array(parameters[p]) <= param_bounds[0][p]) or np.any(np.array(parameters[p]) >= param_bounds[1][p])):
#                 parameters[p] = np.where(np.any(parameters[p] <= param_bounds[0][p]) ,
#                                   np.round(np.random.normal(loc = means[p], scale = stds[p], size = 1), 3),
#                                   parameters[p])
#                 parameters[p] = np.where(np.any(parameters[p] >= param_bounds[1][p]),
#                                   np.round(np.random.normal(loc = means[p], scale = stds[p], size = 1), 3),
#                                   parameters[p])
# =============================================================================
                
                
    return parameters # shape ('npp',)

def simulate_responses(theta = np.array([0,1.6,0.5,1,0.6]), DDM_id = 'angle',nreps = 250):
    """

    Parameters
    ----------
    
    Returns
    -------
    responses : numpy array (with elements of type integer), shape = (ntrials,)
        Array containing the responses simulated by the model for this participant.

    Description
    -----------
    Function to simulate a response on each trial for a given participant with LR = simulation_LR and inverseTemperature = simulation_inverseTemp.
    The design used for data generation for this participant should also be used for parameter estimation for this participant when running the function 'likelihood_estimation'."""
    from ssms.basic_simulators import simulator 
    
    sim_out = simulator(theta = theta,
                        model = DDM_id, 
                        n_samples = nreps)
    
    responses = pd.DataFrame(np.zeros((nreps, 2), 
                        dtype = np.float32), 
                        columns = ['rts', 'choices'])
    
    responses['rts'] = sim_out['rts']
    responses['choices'] = sim_out['choices']
    
    return responses
      
#%% GD
def groupdifference_repetition(inverseTemp_distributions, LR_distributions, npp_per_group,
                               ntrials, start_design, rep, nreps, ncpu, standard_power = False):
    """

    Parameters
    ----------
    inverseTemp_distribution : numpy array, shape = (2,)
        Defines the mean & standard deviation of the normal distribution that will be used to draw the true inverse Temperature values from for each hypothetical participant.
        Mean of the distribution = inverseTemp_distribution[0], standard deviation of the distribution = inverseTemp_distribution[1]
    LR_distributions : numpy array, shape = (2 x 2)
        Defines the mean & standard deviation of the normal distributions that will be used to draw the learning rates from for each hypothetical participant.
        Mean of the distribution for group 0 = LR_distribution[0, 0], standard deviation of the distribution for group 0 = LR_distribution[0, 1].
        Mean of the distribution for group 1 = LR_distribution[1, 0], standard deviation of the distribution for group 1 = LR_distribution[1, 1].
    npp_per_group : integer
        Number of participants in each group that will be used in the parameter recovery analysis.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    start_design : numpy array, shape = (ntrials X 5)
        Design that will be used to simulate data for this repetition and to estimate the parameters as well.
        For more details on this design see function create_design()
    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu:
        The amount of cpu available.

    Returns
    -------
    pValue : float
        Probability to find these recovered learning rate values within the two groups when the two groups would be drawn from the same distribution.
        This probaility is calculated using a two-sample t-test comparing the recovered learning rates for group 0 and group 1.


    Description
    -----------
    Function to execute the group difference statistic once.
    This criterion prescribes that resources are sufficient when a significant group difference is found using the recovered parameters for all participants.
    Thus, the statistic of interest is the p-value returned by a two-sample t-test comparing the recovered parameters of group 0 with the recovered parameters of group 1.
    The group difference is statistically significant when the p-value is smaller than or equal to a specified cut_off (we use a one-sided t-test).
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:
        1. Create 'npp_per_group*2' hypothetical participants for group 0 and group 1 by defining 'npp_per_group*2' parameter sets.
            A parameter set consists of a value for the learning rate and a value for the inverse temperature.
            Two populations are assumed with the following true parameter distributions:
                - learning rates (LRs) group G ~ N(mean, sd) with G = 0 for group 0 and G = 1 for group 1
                    --> mean = LR_distribution[G, 0], sd = LR_distribution[G, 1]
                - inverse temperatures ~ N(mean, sd) for both groups
                    --> mean = Temperature_distribution[0], sd = Temperature_distribution[1]
            npp_per_group parameter sets are created for group 0 and npp_per_group parameter estimates for group 1
        2. Simulate data for each hypothetical participant (thus with each parameter set)
            This is done by simulating responses using the basuc Rescorla-Wagner model (RW-model) with the values of the free parameters = the parameter set of this hypothetical participant.
            This basic RW-model incorporates a delta-learning rule and a softmax choice rule.
            (for details on the basic RW-model see ... (github-link naar ReadME))
        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'recovered parameter values'
            This is done using the Maximum log-Likelihood estimation process in combination with the basic RW-model: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
        4. Calculate the Statistic of interest for this repetition of the parameter recovery analysis.
            The statistic that is calculated here is the p-value associated with the T-statistic which is obtained by a two-sample t-test comparing the recovered LRs for group 0 with the recovered LRs for group 1.
    If this function is repeated a number of times and the value of the Statistic is stored each time, we can evaluate later on the power or probability to meet the proposed parameter recovery criterion (group difference criterion) in a single study.
    """
    if rep == 0:
        t0 = time.time()

    # create array that will contain the final LRestimate for each participant this repetition
    LRestimations = np.empty([2, npp_per_group])
    InvTestimations = np.empty([2, npp_per_group])

    for group in range(2):
        ####PART 1: parameter generation for all participants####
        # Define the True params that will be used for each pp in this rep
        True_LRs =  generate_parameters(mean = LR_distributions[group, 0], std = LR_distributions[group, 1], npp = npp_per_group)
        True_inverseTemps = generate_parameters(mean = inverseTemp_distributions[group, 0], std = inverseTemp_distributions[group, 1], npp = npp_per_group)

        # loop over all pp. to do the data generation and parameter estimation
        for pp in range(npp_per_group):
            ####Part 2: Data simulation for this participant####
            # generate the responses for this participant
            responses = simulate_responses(simulation_LR=True_LRs[pp], simulation_inverseTemp=True_inverseTemps[pp],
                                        design=start_design)
            # fill in the responses of this participant into the start design, in order to use this later in param. estimation
            start_design[:, 2] = responses

            ####Part 3: parameter estimation for this participant####
            # use gradient descent to find the optimal parameters for this participant

            start_params = np.random.uniform(-4.5, 4.5), np.random.uniform(-4.6, 2)
            optimization_output = optimize.minimize(likelihood, start_params, args =(tuple([start_design])),
                                            method = 'Nelder-Mead',
                                            options = {'maxfev':1000, 'xatol':0.01, 'return_all':1})

            estimated_parameters = optimization_output['x']
            estimated_LR = LR_retransformation(estimated_parameters[0])
            estimated_invT = InverseT_retransformation(estimated_parameters[1])

            LRestimations[group, pp] = estimated_LR
            InvTestimations[group, pp] = estimated_invT

    # use two-sided then divide by to, this way we can use the same formula for HPC and non HPC
    Statistic, pValue = stat.ttest_ind(LRestimations[0, :], LRestimations[1, :]) # default: alternative = two-sided
    pValue = pValue/2 # because alternative = less does not exist in scipy version 1.4.0, yet we want a one-sided test
    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * nreps / ncpu
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))

    return Statistic, pValue
#%% EC
def Excorrelation_repetition(inverseTemp_distribution, LR_distribution, true_correlation, npp, ntrials, start_design, rep, nreps, ncpu):
    """

    Parameters
    ----------
    inverseTemp_distribution : numpy array, shape = (2,)
        Defines the mean & standard deviation of the normal distribution that will be used to draw the true inverse Temperature values from for each hypothetical participant.
        Mean of the distribution = inverseTemp_distribution[0], standard deviation of the distribution = inverseTemp_distribution[1]
    LR_distribution : numpy array, shape = (2,)
        Defines the mean & standard deviation of the normal distribution that will be used to draw the true learning rate values from for each hypothetical participant.
        Mean of the distribution = LR_distribution[0], standard deviation of the distribution = LR_distribution[1].
    true_correlation: float
        Defines the hypothesized correlation between the learning rate parameter and an external parameter.
    npp : integer
        Number of participants that will be used in the parameter recovery analysis.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    start_design : numpy array, shape = (ntrials X 5)
        Design that will be used to simulate data for this repetition and to estimate the parameters as well.
        For more details on this design see function create_design()
    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu:
        The amount of cpu available.

    Returns
    -------
    Statistic : float
        The correlation found between the external measure and recovered parameters this repetition.
    pValue : float
        The pvalue for this correlation.
    Stat_true : float
        The pvalue for the correlation between the external measure and true parameters. Indicating the power if estimations would be perfect.

    Description
    -----------
    Function to execute the external correlation statistic once.
    This criterion prescribes that resources are sufficient when: correlation(external measure, recovered learning rates) >= certain cut-off.
    Thus, the statistic of interest is: correlation(measure, recovered learning rates). The correlation is statistically significant when the p-value is smaller than or equal to a specified cut_off.
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:
        1. Create 'npp' hypothetical participants by defining 'npp' parameter sets.
            A parameter set consists of a value for the learning rate and a value for the inverse temperature.
            Additionally, we sample some external measures by considering a multivariate normal distribution for learning rate and theta.
            One population is assumed with the following parameter distributions:
        2. Simulate data for each hypothetical participant (thus each parameter set)
            This is done by simulating responses using the Rescorla-Wagner model (RW-model) with the values of the free parameters = the parameter set of this hypothetical participant.
            This basic RW-model incorporates a delta-learning rule and a softmax choice rule.
            (for details on the basic RW-model see ... (github-link naar ReadME))
        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'recovered parameter values'
            This is done using the Maximum log-Likelihood estimation process in combination with the basic RW-model: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
        4. Calculate the Statistic of interest for this repetition of the analysis.
            The statistic that is calculated here is correlation(measure, recovered learning rates).
    If this function is repeated a number of times and the value of the Statistic is stored each time, we can evaluate later on the power or probability to meet the proposed parameter recovery criterion (the external correlation criterion) in a single study.
    """

    if rep == 0:
        t0 = time.time()

    ####PART 1: parameter generation for all participants####
    # Define the True params that will be used for each pp in this rep
    correlated_values = generate_parameters(mean = LR_distribution[0], std = LR_distribution[1], npp = npp, multivariate = True, corr = true_correlation)
    True_LRs =  correlated_values[:,0]
    True_inverseTemps = generate_parameters(mean = inverseTemp_distribution[0], std = inverseTemp_distribution[1], npp = npp)
    Theta =  correlated_values[:,1]

    # loop over all pp. to do the data generation and parameter estimation
    # create array that will contain the final LRestimate for each participant this repetition
    LRestimations = np.empty(npp)
    invTestimations = np.empty(npp)
    for pp in range(npp):

        ####Part 2: Data simulation for this participant####
        # generate the responses for this participant
        responses = simulate_responses(simulation_LR=True_LRs[pp], simulation_inverseTemp=True_inverseTemps[pp],
                                design=start_design)
        # fill in the responses of this participant into the start design, in order to use this later in param. estimation
        start_design[:, 2] = responses

        ####Part 3: parameter estimation for this participant####
        start_params = np.random.uniform(-4.5, 4.5), np.random.uniform(-4.6, 2)
        optimization_output = optimize.minimize(likelihood, start_params, args =(tuple([start_design])),
                                        method = 'Nelder-Mead',
                                        options = {'maxfev':1000, 'xatol':0.01, 'return_all':1})

        estimated_parameters = optimization_output['x']
        estimated_LR = LR_retransformation(estimated_parameters[0])
        estimated_invT = InverseT_retransformation(estimated_parameters[1])

        LRestimations[pp] = estimated_LR
        invTestimations[pp] = estimated_invT

    ####Part 4: correlation between true & estimated learning rates####
    # if the estimation failed for a certain participant, delete this participant from the correlation estimation for this repetition
    Stat_true = stat.pearsonr(Theta, True_LRs)[1]
    Stat = stat.pearsonr(Theta, LRestimations)
    Statistic = Stat[0]
    pValue = Stat[1]

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    return Statistic, pValue, Stat_true

#%%
