#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:35:59 2023

@author: pieter

Putting together functions to perform parameter estimations
"""
# Import modules
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import softmax, delta_rule, LR_retransformation, InverseT_retransformation

#Avoid warnings
import warnings
warnings.filterwarnings('ignore')
    
#Likelihood function for empirical data
def likelihood(parameter_set, data):
    """

    Parameters
    ----------
    parameter_set : numpy array, shape = (2,)
        Contains the current estimates for each parameter used to calculate the likelihood of the data given this parameter set.
        Contains two values: parameter_set[0] = learning rate, parameter_set[1] = inverse_temperature
    data : csv file name = (ntrials X 4)
        Data that will be used to estimate the likelihood of the data given the current parameter set. The data should contain one row per trial and 4 columns,
        Its columns are: [Trial, Stimulus, Response, Reward].
            The Trial column should contain a value indicating the trial number in increasing order.
            The Stimulus column should contain an integer value of 0 to Nstim-1 for each trial, indicating which stimulus is presented
            The Response column should contain an integer value of 0 to Nresp-1 for each trial, indicating which response was given by the participant
            The Reward column contains a value for the reward that is given to the participant on each trial.
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

    df = pd.read_csv(data)                #Read data
    ntrials = df.shape[0]                 #Extract number of trials
    
    nstim = len(np.unique(df.iloc[:,1]))  #Extract number of stimuli
    nresp = len(np.unique(df.iloc[:,2]))  #Extract number of responses
    
#Prepare the likelihood estimation process: make sure all relevant variables are defined
    # the start values for each stimulus-response pair: a-priori, every response is equally likely and the values are scaled with the max reward
    values = (np.ones((nstim, nresp))/nresp )* np.max(df.iloc[:,3])

#Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0 # this is calculated by summing over trials the log( L(parameter set|response on that trial) )

    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
    #Define the variables that are important for the likelihood estimation process
        stimulus = int(df.iloc[trial,1]) #the stimulus shown on this trial
        response = int(df.iloc[trial,2]) #the response given on this trial
        reward_this_trial = int(df.iloc[trial,3]) #the reward given on this trial

    #Calculate the loglikelihood: log(L(parameter set|response)) = log(P(response|parameter set))
        #select the correct response_values given the stimulus on this trial
        stimulus_weights = values[stimulus, :]

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
    design : csv file name = (ntrials X 4)
        Data that will be used to estimate the likelihood of a response and prediction error given the current parameter set. The data should contain one row per trial and 4 columns,
        Its columns are: [Trial, Stimulus, Response, Reward].
            The Trial column should contain a value indicating the trial number in increasing order.
            The Stimulus column should contain an integer value of 0 to Nstim-1 for each trial, indicating which stimulus is presented
            The Response column should contain an integer value of 0 to Nresp-1 for each trial, indicating which response was given by the participant
            The Reward column contains a value for the reward that is given to the participant on each trial.
    Returns
    -------
    responses : numpy array (with elements of type integer), shape = (ntrials,)
        Array containing the responses simulated by the model for this participant.

    Description
    -----------
    Function to simulate a response on each trial for a given participant with LR = simulation_LR and inverseTemperature = simulation_inverseTemp.
    The design used for data generation for this participant should also be used for parameter estimation for this participant when running the function 'likelihood_estimation'."""

    df = pd.read_csv(data)           #Read data
    ntrials = df.shape[0]            #Extract number of trials
        
    nstim = len(np.unique(df.iloc[:,1]))  #Extract number of stimuli
    nresp = len(np.unique(df.iloc[:,2]))  #Extract number of responses
    
    # the start values for each stimulus-response pair: a-priori, every response is equally likely and the values are scaled with the max reward
    values = (np.ones((nstim, nresp))/nresp )* np.max(df.iloc[:,3])

    # A column list for the output file: we add two columns (Response_likelihood and PE_estimate)
    column_list = ["Trial", "Stimulus", "Response", "Reward", "Response_likelihood", "PE_estimate"]
    simulated_data = pd.DataFrame(columns=column_list)
    
    # trial-loop: generate a response on each trial sequentially
    for trial in range(ntrials):

    #Define the variables you'll need for this trial (take them from the design)
        stimulus = int(df.iloc[trial,1])  #the stimulus shown on this trial
        response = int(df.iloc[trial,2])  #the response given on this trial
        reward_this_trial = int(df.iloc[trial,3])  #the reward on this trial

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
        
        #Store trial data
        simulated_data.loc[trial] = [int(df.iloc[trial,0]) , stimulus, response, reward_this_trial, response_likelihood, PE]

    #Write to output file
    simulated_data.to_csv(data[0:-4]+"_Simulated.csv", columns = column_list, float_format ='%.3f')
    return

if __name__ == '__main__':
    #Extract path to data folder
    directory = sys.argv[1:]
    assert len(directory) == 1
    directory = directory[0]

    #Get a list of files in the folder and filter on csv files that are not previous fitting results
    filelist = os.listdir(directory)
    filtered_filelist = [x for x in filelist if x[-3::]=="csv"]
    filtered_filelist = [x for x in filtered_filelist if x != "Fitting_results.csv"]
    filtered_filelist = [x for x in filtered_filelist if x[-13::] != "Simulated.csv"]

    #Go to that directory
    os.chdir(directory)

    #Define the starting parameters for the likelihood
    start_params = np.random.uniform(-4.5, 4.5), np.random.uniform(-4.6, 2)

    #Define columns for output file
    column_list = ["Subject_ID", "Estimated_LR", "Estimated_InvTemp", "Negative_LogL"]
    estimated_data = pd.DataFrame(columns=column_list)
    
    idx = -1
    for file in filtered_filelist:
        idx +=1
        #Get subject ID
        sub = file.split("_")[2][0:-4]
        
        print("*** Started minimizing negative log likelihood of subject: {} ***\n".format(sub))
        #Minimize negative loglikelihood
        optimization_output = optimize.minimize(likelihood, start_params, args =(tuple([file])), method = 'Nelder-Mead', options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})

        #Get minimum log likelihood and parameter estimations
        LL = optimization_output['fun']
        estimated_parameters = optimization_output['x']
        lr = LR_retransformation(estimated_parameters[0])
        inv_temp = InverseT_retransformation(estimated_parameters[1])
        
        print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
        #Store everything in output file
        estimated_data.loc[idx] = [sub, lr, inv_temp, LL]

        #Simulate with fitted parameters
        simulate_responses(lr, inv_temp, file)
        
        print("Simulated data")

    #Write results of parameter fitting
    estimated_data.to_csv("Fitting_results.csv", columns = column_list, float_format ='%.3f')
    print("End of fitting procedure")
