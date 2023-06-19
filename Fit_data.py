#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:35:59 2023

@author: pieter

Putting together functions to perform parameter estimations
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import softmax, delta_rule, LR_retransformation, InverseT_retransformation

import warnings
warnings.filterwarnings('ignore')
    
    
def likelihood(parameter_set, data):
    """

    Parameters
    ----------
    parameter_set : numpy array, shape = (2,)
        Contains the current estimates for each parameter used to calculate the likelihood of the data given this parameter set.
        Contains two values: parameter_set[0] = learning rate, parameter_set[1] = inverse_temperature
    data : numpy array, shape = (ntrials X 5)
        Data that will be used to estimate the likelihood of the data given the current parameter set. The data should be a numpy array of size (number of trials X 5).
        Its columns are: [stimulus-response mapping rule, stimulus, response, correct response, feedback congruence].
            The stimulus-response mapping rule column should contain a value of 0 or 1 for each trial (= rule 0 or rule 1)
            The stimulus column should contain a value of 0 or 1 for each trial (= stimulus 0 or stimulus 1)
            The response column should contain the simulated responses for the current hypothetical participant (simulated with the function simulate_responses).
            The correct response column should contain which response would be correct on this trial; this depends on the stimulus-response mapping rule and the stimulus for that trial.
            The feedback congruencey column should contain a value of 0 or 1 on each trial with 0 = 'feedback is not in line with the current stimulus-response mapping rule' and 1 = 'feedback is in line with the current stimulus-response mapping rule'.
        Importantly, columns 0, 1, 3 and 4 should be exactly the same as the design matrix used to simulate the responses for this hypothetical participant.

    Returns
    -------
    -summed_logL : float
        The negative summed log likelihood of the data given the current parameter set. This value will be used to select
        the next parameter set that will be evaluated. The goal is to find the most optimal parameters given the data,
        the parameters for which the -summed_logL of all responses is minimal.

    Description
    -----------
    Function to estimate the likelihood of the parameter set under consideration (learning_rate and inverse_temperature) given the data: L(parameter set|data).
    The design is exactly the same as the design used to simulate_data for this hypothetical participant, but now the simulated responses are included as well.
    On each trial: L(parameter set|current response) = P(current response|parameter set). This probability is calculated using the softmax choice rule:
        P(responseX) = exp(value_responseX*inverse_temperature) / (exp(value_response0*inverse_temperature) + exp(value_response1*inverse_temperature)) with X = current response (0 or 1).
        This probability depends on the LR since this defines the value_responseX and on the inverse_temperature since this is part of the softmax function.
    Over trials: summed log likelihood = sum(log(L(parameter set | current response))) with the best fitting parameter set yielding the highest summed logL.
    The function returns -summed_LogL because the optimization function that will be used to find the most likely parameters given the data searches for the minimum value for this likelihood function.
    """
#First retransform the transformedLR to the originalLR
    retransformed_LR = LR_retransformation(parameter_set[0])
    retransformed_invT = InverseT_retransformation(parameter_set[1])

    df = pd.read_csv(data)           #Read data
    ntrials = df.shape[0]                 #Number of trials
    
    nstim = len(np.unique(df.iloc[:,1]))
    nresp = len(np.unique(df.iloc[:,2]))
    
#Prepare the likelihood estimation process: make sure all relevant variables are defined
    # the start values for each stimulus-response pair: these are the same as in the simulate_responses function
    values = np.ones((nstim, nresp))/nresp

#Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0 # this is calculated by summing over trials the log( L(parameter set|response on that trial) )

    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
    #Define the variables that are important for the likelihood estimation process
        stimulus = int(df.iloc[trial,1]) #the stimulus shown on this trial
        response = int(df.iloc[trial,2]) #the response given on this trial
        reward_this_trial = int(df.iloc[trial,3]) #the reward presence or absence on this trial

    #Calculate the loglikelihood: log(L(parameter set|response)) = log(P(response|parameter set))
        #select the correct response_values given the stimulus on this trial
        stimulus_weights = values[stimulus, :]
        #log(P(response|parameter set)) = log(exp(value_responseX*inverse_temperature) / (exp(value_response0*inverse_temperature) + exp(value_response1*inverse_temperature)) with X = current response (0 or 1))
            # which can be simplified to: value_responseX*inverse_temperature - log(exp(value_response0*inverse_temperature) + exp(value_response1*inverse_temperature)) with X = current response
            # used mathematical rule: log( exp(x) / (exp(x)+exp(y)) ) = x - log(exp(x)+exp(y))
        #probabilities = np.exp(loglikelihoods) --> this has to be equal to 1

        # this to ensure no overflows are encountered in the estimation process
        if np.abs(retransformed_invT) > 99:
            # loglikelihoods = np.array([-999, -999])
            summed_logL = -9999
            break
        else: loglikelihoods = np.log(softmax(stimulus_weights, retransformed_invT)) 

        #then select the probability of the actual response given the parameter set
        current_loglikelihood = loglikelihoods[response]

    #Add L(parameter set|current response) to the total log likelihood
        summed_logL = summed_logL + current_loglikelihood
        #Values are updated with current_learning_rate

    #Update the value for the relevant stimulus-response pair using the delta rule
        # (since this influences the probability of the responses on the next trials)
        PE, updated_value = delta_rule(previous_value = values[stimulus, response],
                                                obtained_reward = reward_this_trial,
                                                LR = retransformed_LR)
        values[stimulus, response] = updated_value
    return -summed_logL

def simulate_responses(simulation_LR = 0.5, simulation_inverseTemp = 1, data = "filename"):
    """

    Parameters
    ----------
    simulation_LR : float, optional
        Value for the learning rate parameter that will be used to simulate data for this participant. The default is 0.5.
    simulation_inverseTemp : float, optional
        Value for the inverse temperature parameter that will be used to simulate data for this participant. The default is 1.
    design : numpy array, shape = (ntrials X 5)
        Design that will be used to simulate data for this participant. The design should be a numpy array of size (number of trials X 5).
        Its columns are: [stimulus-response mapping rule, stimulus, response, correct response, feedback congruence].
            The stimulus-response mapping rule column should contain a value of 0 or 1 for each trial (= rule 0 or rule 1)
            The stimulus column should contain a value of 0 or 1 for each trial (= stimulus 0 or stimulus 1)
            The response column should be empty still, data has not yet been generated.
            The correct response column should contain which response would be correct on this trial; this depends on the stimulus-response mapping rule and the stimulus for that trial.
            The feedback congruencey column should contain a value of 0 or 1 on each trial with 0 = 'feedback is not in line with the current stimulus-response mapping rule' and 1 = 'feedback is in line with the current stimulus-response mapping rule'.

    Returns
    -------
    responses : numpy array (with elements of type integer), shape = (ntrials,)
        Array containing the responses simulated by the model for this participant.

    Description
    -----------
    Function to simulate a response on each trial for a given participant with LR = simulation_LR and inverseTemperature = simulation_inverseTemp.
    The design used for data generation for this participant should also be used for parameter estimation for this participant when running the function 'likelihood_estimation'."""

    df = pd.read_csv(data)           #Read data
    ntrials = df.shape[0]    
        
    nstim = len(np.unique(df.iloc[:,1]))
    nresp = len(np.unique(df.iloc[:,2]))
    
    values = np.ones((nstim, nresp))/nresp

    column_list = ["Trial", "Stimulus", "Response", "Reward", "Response_likelihood", "PE_estimate"]
    simulated_data = pd.DataFrame(columns=column_list)
    
    # trial-loop: generate a response on each trial sequentially
    for trial in range(ntrials):

    #Define the variables you'll need for this trial (take them from the design)
        stimulus = int(df.iloc[trial,1])  #the stimulus shown on this trial
        response = int(df.iloc[trial,3])  #the response given on this trial
        reward_this_trial = int(df.iloc[trial,3])  #the reward presence or absence on this trial

    #Simulate the response given by the hypothetical participant. Depending on the value for each response and the inverse_temperature parameter.

        # define which weights are of importance on this trial: depends on which stimulus "appears"
        stimulus_weights = values[stimulus, :]
        # compute probability of each action on this trial (using the weights for each action with the stimulus of this trial)
        response_probabilities = softmax(values = stimulus_weights, inverse_temperature = simulation_inverseTemp)
        # define which action is actually chosen (based on the probabilities)
        response_likelihood = response_probabilities[response]
        #compute the PE and the updated value for this trial (and this stimulus-response pair)
            # to compute the PE & updated value, just work with whether reward was present or not
        PE, updated_value = delta_rule(previous_value=values[stimulus, response],
                                            obtained_reward=reward_this_trial, LR=simulation_LR)
        
        #update the value of the stimulus-response pair that was used this trial
        values[stimulus, response] = updated_value
        
        simulated_data.loc[trial] = [int(df.iloc[trial,0]) , stimulus, response, reward_this_trial, response_likelihood, PE]
    
    simulated_data.to_csv(data, columns = column_list, float_format ='%.3f')
    return

if __name__ == '__main__':
    directory = sys.argv[1:]
    assert len(directory) == 1
    directory = directory[0]
    
    filelist = os.listdir(directory)
    filtered_filelist = [x for x in filelist if x[-3::]=="csv"]
    filtered_filelist = [x for x in filtered_filelist if x != "Fitting_results.csv"]
    
    os.chdir(directory)
    
    start_params = np.random.uniform(-4.5, 4.5), np.random.uniform(-4.6, 2)
    
    column_list = ["Subject_ID", "Estimated_LR", "Estimated_InvTemp", "Negative_LogL"]
    estimated_data = pd.DataFrame(columns=column_list)
    
    idx = -1
    for file in filtered_filelist:
        idx +=1
        sub = file.split("_")[2][0:-4]
        
        print("*** Started minimizing negative log likelihood of subject: {} ***\n".format(sub))
            
        optimization_output = optimize.minimize(likelihood, start_params, args =(tuple([file])), method = 'Nelder-Mead', options = {'maxfev':10000, 'xatol':0.001, 'return_all':1})

        LL = optimization_output['fun']
        estimated_parameters = optimization_output['x']
        lr = LR_retransformation(estimated_parameters[0])
        inv_temp = InverseT_retransformation(estimated_parameters[1])
        
        print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
        estimated_data.loc[idx] = [sub, lr, inv_temp, LL]
        
        simulate_responses(lr, inv_temp, file)
        
        print("Simulated data")
    
    estimated_data.to_csv("Fitting_results.csv", columns = column_list, float_format ='%.3f')
    print("End of fitting procedure")
