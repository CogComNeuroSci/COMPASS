
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:09:53 2021
@author: Maud
"""
import ssms
import numpy as np
import pandas as pd
import os, sys, time
import math
from scipy import optimize
from scipy import stats as stat
import matplotlib.pyplot as plt
from Likelihoods import neg_likelihood
from ParameterEstimation import MLE

#This is to avoid warnings being printed to the terminal window
import warnings
warnings.filterwarnings('ignore')


def generate_parameters_DDM(means,stds, param_bounds, npp = 150, multivariate = False, par_ind = 0 , 
                        corr = False):
    """
    generate_parameters_DDM
    ----------
    means : 1 * n array 
        means of parameters in an array. n: number of parameters
        if 
    stds : 1 * n array 
        The standard deviation of the normal distribution from which parameters are drawn. The default is 0.1.
    param_bounds: 2 * n array
        min and max of parameters 
    npp: int
        sample size of participants and parameters      
    multivariate: boolean, optional
        Put to true for the external correlation criterion such that values are drawn from multivariate normal distribution. The default is False.
    par_ind: int, optional (only used when multivariate = True)
        the index of parameter which is hypothetically correlated with an external measure

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
        mean  = means[par_ind]
        std = stds[par_ind]
        # draw 'npp' values from multivariate normal distribution with mean 'mean', standard deviation 'std' and correlation 'cor'
        parameters = np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), npp),3)
        # while-loop: ensure no learning rate parameters get a value smaller than or equal to 0
        while max(parameters[:,0])>param_bounds[1][par_ind] or min(parameters[:,0])<param_bounds[0][par_ind]:
            outerID = np.logical_or(parameters[:,0] <= param_bounds[0][par_ind],parameters[:,0] >= param_bounds[1][par_ind])
            parameters[outerID] = np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), size = sum(outerID)),3)

    else:
        # sample all parameters 
        parameters = np.zeros((npp, len(means)))

            
        for p in range(len(means)):
            # draw 'npp' values from normal distribution with mean 'mean' and standard deviation 'std'
            parameters[:,p] = np.round(np.random.normal(loc = means[p], scale = stds[p], size = npp), 3)
            
            while max(parameters[:,p])>param_bounds[1][p] or min(parameters[:,p])<param_bounds[0][p]:
                outerID = np.logical_or(parameters[:,p] <= param_bounds[0][p],parameters[:,p] >= param_bounds[1][p])
                parameters[outerID,p] = np.round(np.random.normal(loc = means[p], scale = stds[p], size = sum(outerID)), 3)
                
            # while-loop: ensure no parameters get a value smaller than or equal to 0
# =============================================================================
#             while (np.any(np.array(parameters[p]) <= param_bounds[][p]) or np.any(np.array(parameters[p]) >= param_bounds[1][p])):
#                 parameters[p] s= np.where(np.any(parameters[p] <= param_bounds[0][p]) ,
#                                   np.round(np.random.normal(loc = means[p], scale = stds[p], size = 1), 3),
#                                   parameters[p])
#                 parameters[p] = np.where(np.any(parameters[p] >= param_bounds[1][p]),
#                                   np.round(np.random.normal(loc = means[p], scale = stds[p], size = 1), 3),
#                                   parameters[p])
# =============================================================================
            # parameters = pd.DataFrame(parameters, dtype = np.float32)
                
    return parameters # shape ('npp',)
def simulate_responses_DDM(theta = np.array([0,1.6,0.5,1,0.6]), DDM_id = 'angle',n_samples = 250):
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
                        n_samples = n_samples)
    
    responses = pd.DataFrame(np.zeros((n_samples, 2), 
                        dtype = np.float32), 
                        columns = ['rts', 'choices'])
    
    responses['rts'] = sim_out['rts']
    responses['choices'] = sim_out['choices']
    
    return responses

def Incorrelation_repetition_DDM(means,stds , 
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
    param_bounds_Opti = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2 


    for pp in range(npp): # loop for participants
        ACC = 0
        ####Part 2: Data simulation for this participant####
        while ACC <= 0.50 or ACC >= 0.95 or RT >= 10:
        # generate the responses for this participant

            True_Par.iloc[pp,:] = generate_parameters_DDM(means = means, stds = stds, 
                                     param_bounds = param_bounds, npp = 1)



            responses = simulate_responses_DDM(np.array(True_Par.iloc[pp,:]),DDM_id,ntrials)
            responses = np.array(responses['rts'] * responses['choices'])
            # validation of parameters
            
            ACC = np.mean( responses*True_Par.iloc[pp,0] > 0)            
            RT = np.mean(np.abs(responses))
            
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10: 
                waste_counter = waste_counter+1
         
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
        print("Sample: {}/{}, Statistic of parameter {}: r = {}".format(rep,nreps,ssms.config.model_config[DDM_id]['params'][p],Statistic[0,p]))

 
    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    
    return Statistic, True_Par, Esti_Par, ACC_out, RT_out
def Excorrelation_repetition_DDM(means,stds , param_bounds, par_ind,DDM_id,true_correlation, 
                                 npp, ntrials,rep, nreps, ncpu=6, method = 'Nelder-Mead'):
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
            A parameter set consists of parameters from DDM model of interest.
            Additionally, we sample some external measures by considering a multivariate normal distribution for parameters.
            One population is assumed with the following parameter distributions:

        2. Simulate data for each hypothetical participant (thus each parameter set)
            This is done by simulating responses using the SSMS package with the values of the free parameters = the parameter set of this hypothetical participant.

        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'recovered parameter values'
            This is done using the Maximum log-Likelihood estimation process in combination with the DDM model: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
        
        4. Calculate the Statistic of interest for this repetition of the analysis.
            The statistic that is calculated here is correlation(measure, recovered learning rates).
    If this function is repeated a number of times and the value of the Statistic is stored each time, we can evaluate later on the power or probability to meet the proposed parameter recovery criterion (the external correlation criterion) in a single study.
    """
   
    if rep == 0:
        t0 = time.time()

    ####PART 1: parameter generation for all participants####
    # Define the True params that will be used for each pp in this rep
    True_Par = pd.DataFrame(np.empty((npp,len(means))))
    True_Par.columns = ssms.config.model_config[DDM_id]['params']

    correlated_values = generate_parameters_DDM(means, stds, param_bounds, npp = npp, 
                                                multivariate = True, par_ind = par_ind,corr = true_correlation)
    True_Par.iloc[:,par_ind] =  correlated_values[:,0]


    Theta =  correlated_values[:,1]

    # loop over all pp. to do the data generation and parameter estimation
    # create array that will contain the final LRestimate for each participant this repetition

    Esti_Par = pd.DataFrame(np.empty((npp,len(means))))
    Esti_Par.columns = ssms.config.model_config[DDM_id]['params']

    # param_bounds_Opti = param_bounds
    # param_bounds_Opti[:,1] = param_bounds[:,1]*2 
    param_bounds_Opti = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2 

    Col_UncPar = np.delete(range(len(means)),par_ind )
    waste_counter = 0
    for pp in range(npp):
        ACC = 0
        ####Part 2: Data simulation for this participant####
        while ACC <= 0.50 or ACC >= 0.95 or RT >= 10:
        # generate the responses for this participant

            True_Par.iloc[pp, Col_UncPar] = generate_parameters_DDM(means = means[Col_UncPar], stds = stds[Col_UncPar], param_bounds = param_bounds, npp = 1)
    
            responses = simulate_responses_DDM(theta=True_Par.iloc[pp,:].values, DDM_id = DDM_id, n_samples = ntrials)
            # fill in the responses of this participant into the start design, in order to use this later in param. estimation
            responses = np.array(responses['rts'] * responses['choices'])
                        # validation of parameters
                
            ACC = np.mean( responses * True_Par.iloc[pp,0] > 0)            
            RT = np.mean(np.abs(responses))
            
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10: 
                waste_counter = waste_counter+1



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
    True_r = stat.pearsonr(Theta, True_Par.iloc[:,par_ind])[0]
    True_pValue = stat.pearsonr(Theta, True_Par.iloc[:,par_ind])[1]
    Stat = stat.pearsonr(Theta, Esti_Par.iloc[:,par_ind])
    Esti_r = Stat[0]
    Esti_pValue = Stat[1]

    print('sampel: {}/{}, statistics: r = {:.3f}, p = {:.3f}'.format(rep,nreps,Esti_r,Esti_pValue))

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    return Esti_r, Esti_pValue, True_r, True_pValue
def Groupdifference_repetition_DDM(means_g1, stds_g1,means_g2, stds_g2,DDM_id, par_ind,param_bounds,
                                   npp_per_group, ntrials, rep, nreps, ncpu, standard_power = False):
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
    # LRestimations = np.empty([2, npp_per_group])
    # InvTestimations = np.empty([2, npp_per_group])
    

    cn = ["group",]+ssms.config.model_config[DDM_id]['params']

    True_Par = pd.DataFrame(np.empty([npp_per_group*2,len(means_g1)+1]))
    True_Par.columns = cn   
    Esti_Par = pd.DataFrame(np.empty((npp_per_group*2,len(means_g1)+1)))
    Esti_Par.columns = cn
    ACC_out = np.empty((npp_per_group*2,2))
    RT_out = np.empty((npp_per_group*2,2))

    param_bounds_Opti =  np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2  # rescale parameter a
    waste_counter = 0



        # loop over all pp. to do the data generation and parameter estimation
    for pp in range(npp_per_group*2):
        if pp <= npp_per_group-1:
            group = 1
            means = means_g1
            stds = stds_g1
        else:
            group = 2
            means = means_g2
            stds = stds_g2
        
        ACC = 0
        ####Part 2: Data simulation for this participant####
        while ACC <= 0.50 or ACC >= 0.95 or RT >= 10:
            # generate the responses for this participant
            True_Par.iloc[pp,0] = group
            True_Par.iloc[pp, 1:] = generate_parameters_DDM(means = means, stds = stds,
                                                                param_bounds = param_bounds, npp = 1)
    
            responses = simulate_responses_DDM(theta=True_Par.iloc[pp,1:].values, DDM_id = DDM_id, n_samples = ntrials)
            # fill in the responses of this participant into the start design, in order to use this later in param. estimation
            responses = np.array(responses['rts'] * responses['choices'])
                        # validation of parameters
            
            ACC = np.mean( responses * True_Par['v'][pp] > 0)     # [pp,1]:index of v, drfit rate       
            RT = np.mean(np.abs(responses))
            
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10: 
                waste_counter = waste_counter+1

        ACC_out[pp,group-1] = ACC
        RT_out[pp,group-1] = RT

        ####Part 3: parameter estimation for this participant####
        fun = neg_likelihood
        arg = (responses,DDM_id)
        method ='Nelder-Mead'
        # method = "Nelder-Mead"  or method=="Brute"
    # re-scaling parameter a
        Esti_Par.iloc[pp,0] = group 
        Esti_Par.iloc[pp,1:] = MLE(fun,arg,param_bounds_Opti,method,show = 0)     

        # print(Esti_Par[pp],neg_likelihood(Esti_Par[pp],arg)) 
        # print(np.array(True_Par.loc[pp]),neg_likelihood(True_Par.loc[pp],arg))

    Esti_Par['a'] = Esti_Par['a']/2

    # use two-sided then divide by two, this way we can use the same formula for HPC and non HPC

    g1_df = Esti_Par[Esti_Par['group'] == 1]
    g2_df = Esti_Par[Esti_Par['group'] == 2]
    # default: alternative = two-sided
    Statistic, pValue = stat.ttest_ind(g1_df.iloc[:,par_ind+1], g2_df.iloc[:,par_ind+1]) 
    # because alternative = less does not exist in scipy version 1.4.0, yet we want a one-sided test
    pValue = pValue/2 

    print('sampel: {}/{}, statistics: t = {:.3f}, p = {:.3f}'.format(rep,nreps,Statistic,pValue))

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * nreps / ncpu
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))

    return Statistic, pValue
