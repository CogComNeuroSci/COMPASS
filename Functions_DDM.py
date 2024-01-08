
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
    Parameters
    ----------
    means : 1 * n array 
        Means of parameters in an array. n: number of parameters
    stds : 1 * n array 
        The standard deviation of the normal distribution from which parameters are drawn. The default is 0.1.
    param_bounds: 2 * n array
        Min and max of parameters 
    npp: int
        Sample size of participants and parameters      
    multivariate: boolean, optional
        Put to true for the external correlation criterion such that values are drawn from multivariate normal distribution. The default is False.
    par_ind: int, optional (only used when multivariate = True)
        The index of parameter which is hypothetically correlated with an external measure
    corr: boolean or float, optional
        The correlation for the external correlation criterion. For other criterions this is ignored. The default is False.
    Returns
    -------
    parameters : npp * len(means) array or npp * 2 array (when multivariate = True)
        Array with shape ('size',) containing the parameters drawn from the normal distribution.
        When multivariate = True, it returns correlated parameters generated from a multivariate normal distribution. 
        Its covariance matrix is defined by the assumed correlation coeffient and the std of the parameter of interest
        The column of 0 index is the correlated parameters. The column of of 1 index is the external measure assumly correlated to the parameters.

    Description
    -----------
    Function to draw 'npp' parameters from a normal distribution with mean 'mean' and standard deviation 'std'.
    Function is used to generate parameters for each participant.
    When the criterion is external correlation, the parameter of interest and the external measure are drawn from a multivariate normal distribution.
    Here, the correlation is specified in the covariance matrix."""

    if multivariate:
        mean  = means[par_ind]
        std = stds[par_ind]
        # draw 'npp' values from multivariate normal distribution with mean 'mean', standard deviation 'std' and correlation 'cor'
        parameters = np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), npp)
        # while-loop: ensure no learning rate parameters get a value smaller than or equal to 0
        while max(parameters[:,0])>param_bounds[1][par_ind] or min(parameters[:,0])<param_bounds[0][par_ind]:
            outerID = np.logical_or(parameters[:,0] <= param_bounds[0][par_ind],parameters[:,0] >= param_bounds[1][par_ind])
            parameters[outerID] = np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), size = sum(outerID))

    else:
        # sample all parameters 
        parameters = np.zeros((npp, len(means)))

            
        for p in range(len(means)):
            # draw 'npp' values from normal distribution with mean 'mean' and standard deviation 'std'
            parameters[:,p] = np.random.normal(loc = means[p], scale = stds[p], size = npp)
            
            while max(parameters[:,p])>param_bounds[1][p] or min(parameters[:,p])<param_bounds[0][p]:
                outerID = np.logical_or(parameters[:,p] <= param_bounds[0][p],parameters[:,p] >= param_bounds[1][p])
                parameters[outerID,p] = np.random.normal(loc = means[p], scale = stds[p], size = sum(outerID))
                
                
    return parameters # shape ('npp',)
def simulate_responses_DDM(theta = np.array([0,1.6,0.5,1,0.6]), DDM_id = 'angle',n_samples = 250):
    """

    simulate_responses_DDM
    ----------
        theta : 1 * n array,  n: number of parameters 
            Parameters in an array to generate responses from ssms package.
        DDM_id: string
            Index of DDM model which should be matched with ssms package
        n_samples: int
            Number of trials that will be simulated

    
    Returns
    -------
    responses : Dataframe shape = (ntrials,2)
        Dataframe containing the responses simulated by the model for this participant, with 2 columns.
        The 1st col: RTs; the 2nd col: choices denoted by -1 or 1

    Description
    -----------
    Function to simulate a response on each trial for one given participant with thetas, which contains input parameters with the same order as in ssms.
    """
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
    means: 1 * n array
        Means of distribution from which true parameters are sampled
    stds: 1 * n array
        Stds of distribution from which true parameters are sampled
    param_bounds: 2 * n array
        Range of parameters, the 0th row corresponds to lower bound, the 1st corresponds to upper bound
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    DDM_id: string
        Index of DDM model which should be matched with SSMS package
    method: string
        Method of optimization, either "Nelder-Mead" or "Brute"
    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu:
        The amount of cpu available.

    Returns
    -------
    Statistic_Proficienct : array, shape = (number of parameters + 2,), saved in output file
        Index 0 to (number of parameters-1): The correlation found between the true and recovered parameters this repetition.
        Index end-1: average ACC of all participants. ACC was calculated by whether the sign of drift rate and final decision are of the same sign.
        Index end: average RT of all participants.
    True_Par: array, not saved in output file
        True parameters sampled from the distributions defined in the input file.
    Esti_Par: array, not saved in output file
        Estimated parameters of generated behavioral data.

    Description
    -----------
    Function to execute the parameter recovery analysis (Internal correlation criterion) once.
    This criterion prescribes that resources are sufficient when: correlation(true parameters, recovered parameters) >= certain cut-off.
    Thus, the statistic of interest is: correlation(ttrue parameters, recovered parameters). This statistic is returned for execution of this function (thus for each repetition).
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:
        1. Create ONE hypothetical participants by defining a parameter set.
            A parameter set consists of values required by SSMS package. Parameters are sampled from the Gaussian distribution defined in the input file.

        2. Simulate data for ONE hypothetical participant (thus one parameter set).
            This is done by simulating responses using the DDM model from SSMS package with the values of the free parameters = the parameter set of this hypothetical participant.

        3. Test the performance of this hypothetical participant.
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10, then resample the parameter set from the defined distributions.

        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'Esti_Par'.
            This is done using the Maximum log-Likelihood estimation process: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
            For standard DDM: analytical likelihood function is used.

        4. Back to step 1. for next hypothetical participants, until all npp participants were finished.

        5. Calculate the Statistic of interest for this repetition of the parameter recovery analysis.
            The statistic that is calculated here is correlation(true parameters, recovered parameters).
        6. Average ACC and RT were calculated and saved into output file along with Statistics.
    If this function is repeated a number of times and the value of the Statistic is stored each time, we can evaluate later on the power or probability to meet the proposed parameter recovery criterion (the internal correlation criterion) in a single study.
    
    NOTE: RESCALING BOUNDARY PARAMETER "A"
    As the parameter a was unmatched scaled in wfpt likelihood functions, this function includes corrected scales of this parameter explicitly.
    It is achieved by: 
    (1) double optimization range of 'a', i.e., param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2  
    (2) take 1/2 of estimated parameter "a"
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
    if DDM_id == "ddm":
        param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2  


    for pp in range(npp): # loop for participants
        ACC = 0
        ####Part 2: Data simulation for this participant####
        while ACC <= 0.5 or ACC >= 0.95 or RT >= 10:
        # generate the responses for this participant

            True_Par.iloc[pp,:] = generate_parameters_DDM(means = means, stds = stds, 
                                     param_bounds = param_bounds, npp = 1)



            responses = simulate_responses_DDM(np.array(True_Par.iloc[pp,:]),DDM_id,ntrials)
            responses = np.array(responses['rts'] * responses['choices'])
            # validation of parameters
            
            ACC = np.mean(responses * True_Par.iloc[pp,0] > 0)            
            RT = np.mean(np.abs(responses))
            
            if ACC <= 0.5 or ACC >= 0.95 or RT >= 10: 
                waste_counter = waste_counter+1
         
        ACC_out[pp] = ACC
        RT_out[pp] = RT

        ####Part 3: parameter estimation for this participant####
        fun = neg_likelihood
        arg = (responses,DDM_id)
         # method = "Nelder-Mead"  or method=="Brute"

        Esti_Par.iloc[pp,:] = MLE(fun,arg,param_bounds_Opti,method,show = 0)     

        # print(Esti_Par[pp],neg_likelihood(Esti_Par[pp],arg)) 
        # print(np.array(True_Par.loc[pp]),neg_likelihood(True_Par.loc[pp],arg))

    # re-scaling parameter a
    Esti_Par['a'] = Esti_Par['a']/2
    ####Part 4: correlation between true & estimated learning rates####
    # if the estimation failed for a certain participant, delete this participant from the correlation estimation for this repetition
    ACC_average = round(float(sum(ACC_out))/len(ACC_out),3)
    RT_average = round(float(sum(RT_out))/len(RT_out),3)
    Statistic = np.empty((1,len(means)))
    for p in range(len(means)):
        Statistic[0,p] = np.round(np.corrcoef(True_Par.iloc[:,p], Esti_Par.iloc[:,p])[0,1], 3)
        print("Sample: {}/{}, Statistic of parameter {}: r = {}".format(rep,nreps,ssms.config.model_config[DDM_id]['params'][p],Statistic[0,p]))
    
    Statistic_Proficienct = np.append(Statistic,ACC_average)
    Statistic_Proficienct = np.append(Statistic_Proficienct,RT_average)
 
    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    
    return Statistic_Proficienct, True_Par, Esti_Par
def Excorrelation_repetition_DDM(means,stds , param_bounds, par_ind,DDM_id,true_correlation, 
                                 npp, ntrials,rep, nreps, ncpu=6, method = 'Nelder-Mead'):
    """

    Parameters
    ----------
    means: 1 * n array
        Means of distribution from which true parameters are sampled
    stds: 1 * n array
        Stds of distribution from which true parameters are sampled
    param_bounds: 2 * n array
        range of parameters, the 0th row corresponds to lower bound, the 1st corresponds to upper bound
    par_ind: int
        The index of parameter of interest, start
    DDM_id: string
        Index of DDM model which should be matched with SSMS package
    true_correlation:
        Defines the hypothesized correlation between the parameter of interest and an external parameter.
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu: int
        The amount of cpu available.
    method: string
        Method of optimization, either "Nelder-Mead" or "Brute"

    Returns 
    -------
    Esti_r: float
        The Pearson correlation coeffient of the estimated parameters and the true external external measures
    Esti_pValue: float
        The p-value of Esti_r
    True_r: float
        The Pearson correlation coeffient of the generated true parameters and the true external external measures
    True_pValue: float
        The p-value of True_pValue
    ACC_average:
        Average ACC of all participants. ACC was calculated by whether the sign of drift rate and final decision are of the same sign. 
    RT_average:
        Average RT of all participants. 
    Description
    -----------
    Function to execute the external correlation statistic once.
    This criterion prescribes that resources are sufficient when: correlation(external measure, recovered parameter) >= certain cut-off.
    Thus, the statistic of interest is: correlation(measure, recovered parameter). The correlation is statistically significant when the p-value is smaller than or equal to a specified cut_off.
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:
        1. Generate correlated true parameter of interest and external measure based on defined mean, std of that parameter distribution and true correlation.
            npp of parameters and external measure pairs will be generated and fixed through out simulation.
            This is designed to enable resample other parameters, which are not assumed to be correlated with an external measure, to achienve a validate ACC and RT
            but remain the correlation between the parameter of interest and external measure.

        2. Sample other parameters required for SSMS package models for one hypothetical participant, which are not assumed to be correlated with an external measure. 
            (Start looping for npp hypothetical participants.)

        3. Given all parameters requireed, simulate behavioral data for this hypothetical participant. 
            Validate behavioral performance of this participant. If ACC <= 0.50 or ACC >= 0.95 or RT >= 10, then go back to step 2.

        4. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'Esti_Par'.
            This is done using the Maximum log-Likelihood estimation process: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
            For standard DDM: analytical likelihood function is used.
        
        5. Back to step 2. for next hypothetical participants, until all npp participants were finished.

        6. Calculate and return the statistics for this repetition of the analysis, including:
            correlation(external measure, recovered parameter of interest), and its p-value;
            correlation(external measure, true parameter of interest), and its p-value;
            average ACC and RT
    
    NOTE: RESCALING BOUNDARY PARAMETER "A"
    As the parameter a was unmatched scaled in wfpt likelihood functions, this function includes corrected scales of this parameter explicitly.
    It is achieved by: 
    (1) double optimization range of 'a', i.e., param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2  
    (2) take 1/2 of estimated parameter "a"

    """
   
    if rep == 0:
        t0 = time.time()

    ####PART 1: parameter generation for all participants####
    # Define the True params that will be used for each pp in this rep
    True_Par = pd.DataFrame(np.empty((npp,len(means))))
    True_Par.columns = ssms.config.model_config[DDM_id]['params']
    # Generate the correlated parameter and the external measure as true parameters
    correlated_values = generate_parameters_DDM(means, stds, param_bounds, npp = npp, 
                                                multivariate = True, par_ind = par_ind,corr = true_correlation)
    True_Par.iloc[:,par_ind] =  correlated_values[:,0]
    Theta =  correlated_values[:,1] # the external measure

    Esti_Par = pd.DataFrame(np.empty((npp,len(means))))
    Esti_Par.columns = ssms.config.model_config[DDM_id]['params']

    # rescale optimizing range of parameter 'a'
    param_bounds_Opti = np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    if DDM_id =="ddm":
        param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2 

    Col_UncPar = np.delete(range(len(means)),par_ind )
    waste_counter = 0
    ACC_out = np.empty((npp,1))
    RT_out = np.empty((npp,1))
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

        ACC_out[pp] = ACC
        RT_out[pp] = RT

        ####Part 3: parameter estimation for this participant####
        fun = neg_likelihood
        arg = (responses,DDM_id)
        # method = "Nelder-Mead"  or method=="Brute"
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

    print('sample: {}/{}, statistics: r = {:.3f}, p = {:.3f}'.format(rep,nreps,Esti_r,Esti_pValue))

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    ACC_average = round(float(sum(ACC_out))/len(ACC_out),3)
    RT_average = round(float(sum(RT_out))/len(RT_out),3)

    return Esti_r, Esti_pValue, True_r, True_pValue, ACC_average, RT_average
def Groupdifference_repetition_DDM(means_g1, stds_g1,means_g2, stds_g2,DDM_id, par_ind,param_bounds,
                                   npp_per_group, ntrials, rep, nreps, ncpu, method = 'Nelder-Mead'):
    """

    Parameters
    ----------
    means_g1: array
        Means of the distribution from which samples true parameters of group 1
    means_g2: array
        Means of the distribution from which samples true parameters of group 2
    stds_g1: array
        Stds of the distribution from which samples true parameters of group 1
    stds_g2: array
        Stds of the distribution from which samples true parameters of group 2
    DDM_id: string
        Index of DDM model which should be matched with ssms package
    par_ind: int
        The index of parameter of interest, start from 0
    param_bounds: 2 * n array
        Range of parameters, the 0th row corresponds to lower bound, the 1st corresponds to upper bound
    npp_per_group : integer
        Number of participants in each group that will be used in the parameter recovery analysis.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    rep : integer
        Which repetition of the power estimation process is being executed.
    nreps: integer
        The total amount of repetitions.
    ncpu:
        The amount of cpu available.
    method: string
        Method of optimization, either "Nelder-Mead" or "Brute"
    

    Returns
    -------
    Statistic: float
        t-value of a two-sample t-test comparing the recovered parameters of interest for group 1 and group 2.
    pValue: float
        p-value of the t-value.
    ACC_g1: float
        average ACC of group 1
    ACC_g2: float
        average ACC of group 2
    RT_g1: float
        average RT of group 1
    RT_g2: float
        average RT of group 2


    Description
    -----------
    Function to execute the group difference statistic once.
    This criterion prescribes that resources are sufficient when a significant group difference is found using the recovered parameters for all participants.
    Thus, the statistic of interest is the p-value returned by a two-sample t-test comparing the recovered parameters of group 1 with the recovered parameters of group 2.
    The group difference is statistically significant when the p-value is smaller than or equal to a specified cut_off (we use a one-sided t-test).
    In order to calculate the statistic the parameter recovery analysis has to be completed. This analysis consists of several steps:

        1. Create ONE hypothetical participants by defining a parameter set. 
            A parameter set consists of values required by SSMS package. Parameters are sampled from the Gaussian distribution defined in the input file.
            If number of samples exceeds npp_per_group, then sampple true parameters from the group 2 distribution.

        2. Simulate data for ONE hypothetical participant (thus one parameter set).
            This is done by simulating responses using the DDM model from SSMS package with the values of the free parameters = the parameter set of this hypothetical participant.

        3. Test the performance of this hypothetical participant.
            if ACC <= 0.50 or ACC >= 0.95 or RT >= 10, then resample the parameter set from the defined distributions.

        3. Estimate the best fitting parameter set given the simulated data: best fitting parameter set = 'Esti_Par'.
            This is done using the Maximum log-Likelihood estimation process: iteratively estimating the log-likelihood of different parameter values given the data.
            The parameter set with the highest log-likelihood given the data is selected. For more details on the likelihood estimation process see function 'likelihood'.
            For standard DDM: analytical likelihood function is used.

        4. Back to step 1. for next hypothetical participants, until all "npp_per_group*2" participants were finished.

        5. Calculate the Statistic of interest for this repetition of the parameter recovery analysis.
            The statistic that is calculated here is the p-value associated with the T-statistic which is obtained by a two-sample t-test comparing the recovered parameter of interest for group 1 and group 2.

    NOTE: RESCALING BOUNDARY PARAMETER "A"
    As the parameter a was unmatched scaled in wfpt likelihood functions, this function includes corrected scales of this parameter explicitly.
    It is achieved by: 
    (1) double optimization range of 'a', i.e., param_bounds_Opti[:,1] = param_bounds_Opti[:,1]*2  
    (2) take 1/2 of estimated parameter "a"
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
    if DDM_id == "ddm":
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
        Esti_Par.iloc[pp,0] = group 
        Esti_Par.iloc[pp,1:] = MLE(fun,arg,param_bounds_Opti,method,show = 0)     

        # print(Esti_Par[pp],neg_likelihood(Esti_Par[pp],arg)) 
        # print(np.array(True_Par.loc[pp]),neg_likelihood(True_Par.loc[pp],arg))
    # re-scaling parameter a
    Esti_Par['a'] = Esti_Par['a']/2

    # use two-sided then divide by two, this way we can use the same formula for HPC and non HPC

    g1_df = Esti_Par[Esti_Par['group'] == 1]
    g2_df = Esti_Par[Esti_Par['group'] == 2]
    # default: alternative = two-sided
    Statistic, pValue = stat.ttest_ind(g1_df.iloc[:,par_ind+1], g2_df.iloc[:,par_ind+1]) 
    # because alternative = less does not exist in scipy version 1.4.0, yet we want a one-sided test
    pValue = pValue/2 

    print('Sampel: {}/{}, statistics: t = {:.3f}, p = {:.3f}'.format(rep,nreps,Statistic,pValue))

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * nreps / ncpu
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    ACC_g1 = np.mean(ACC_out, axis=0)[0]
    ACC_g2 = np.mean(ACC_out, axis=0)[1]
    RT_g1 = np.mean(RT_out, axis=0)[0]
    RT_g2 = np.mean(RT_out, axis=0)[1]

    return Statistic, pValue, ACC_g1,ACC_g2,RT_g1,RT_g2