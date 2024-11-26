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

#%% RL functions
#%%% Transformating parameters to infinite range
def LR_retransformation_RL(transformed_LR = 1.3):
    original_LR = np.exp(transformed_LR)/(1+np.exp(transformed_LR))
    return original_LR

def InverseT_retransformation_RL(transformed_InverseT = 3):
    original_InverseT = np.exp(transformed_InverseT)
    return original_InverseT

#%%% Functions that are used within other functions in this 'functions' script
def generate_parameters_RL(mean = 0.5, std = 0.1, npp = 1, multivariate = False, corr = False):
    """
    Parameters
    ----------
    mean : float or int, optional
        The mean value of the normal distribution from which parameters are drawn. The default is 0.5.
    std : float or int, optional
        The standard deviation of the normal distribution from which parameters are drawn. The default is 0.1.
    size : float, optional
        The number of parameters that are drawn from the normal distribution. The default is 1.
    multivariate: boolean, optional
        Put to true for the external correlation criterion such that values are drawn from multivariate normal distribution. The default is False.
    corr: boolean or float, optional
        The correlation for the external correlation criterion. For other criterions this is ignored. The default is False.
    Returns
    -------
    parameters : numpy array
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
        parameters =np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), npp),3)
        # while-loop: ensure no learning rate parameters get a value smaller than or equal to 0
        while np.any(parameters[:,0] <= 0):
            to_replace = np.where(parameters[:,0] <= 0)[0]
            parameters[to_replace, :] = np.round(np.random.multivariate_normal([mean, 0], np.array([[corr*std, std], [std, corr*std]]), len(to_replace)),3)
    else:
        # draw 'npp' values from normal distribution with mean 'mean' and standard deviation 'std'
        parameters = np.round(np.random.normal(loc = mean, scale = std, size = npp), 3)
        # while-loop: ensure no parameters get a value smaller than or equal to 0
        while np.any(parameters <= 0):
            parameters = np.where(parameters <= 0,
                              np.round(np.random.normal(loc = mean, scale = std, size = 1), 3),
                              parameters)
    return parameters # shape ('npp',)

def softmax_RL(values = np.array([0.5, 0.5]), inverse_temperature = 1):
    """
    Parameters
    ----------
    values : numpy array, optional
        The activation level for the possible responses.
        The default is np.array([0.5, 0.5]).
    inverse_temperature : TYPE, optional
        The amount of randomness in the responses; lower inverse temperature = more randomness.
        The default is 1.

    Returns
    -------
    response_probabilities : numpy array
        The probability with which each response option will be chosen (sum of the probabilities = 1).

    Description
    -----------
    The softmax function returns the probability of choosing each response option. In general, the option with the highest value has the highest probability to be chosen.
    The inverse temperature parameter captures the weight given to the value difference between the two response options. With a low inverse temperature,
    more weight is given to the option with the highest value ('exploiting the best option'). A higher inverse tempererature relies more on 'exploration in order to find better options',
    thus increases the probability of choosing the lower value option.
    The softmax function is: probability(response X) = exp(value_responseX*inverse_temperature) / (exp(value_response0*inverse_temperature) + exp(value_response1*inverse_temperature)) with X = 0 or x = 1."""
    # softmax function
    response_probabilities = np.exp(values*inverse_temperature) / np.sum((np.exp(values*inverse_temperature)))
    return response_probabilities

def choose_response_RL(response_probabilities = np.array([0.5, 0.5])):
    """

    Parameters
    ----------
    response_probabilities : numpy array, optional
        The probabilities for choosing each of the two responses. The default is np.array([0.5, 0.5]).

    Returns
    -------
    response : integer (0 or 1)
        Which of the two responses is actually chosen: response 0 or response 1.


    Description
    -----------
    Function to actually choose response 0 or 1. This function randomly generates a value between 0 and 1.
    If this value is smaller than or equal to the probability to choose response 1 (response_probabilities[1]), then response 1 is chosen.
    If this value is larger than the probability to choose response 1, response 0 is chosen."""
    response = (np.random.random() <= response_probabilities[1])*1
    #sortresps = np.argsort(response_probabilities)[::-1]
    #response = sortresps[np.where(np.random.random() <= np.cumsum(response_probabilities[sortresps]))[0][0]]
    return response

def delta_rule_RL(previous_value = 0.0, obtained_reward = 1.0, LR = 0.1):
    """

    Parameters
    ----------
    previous_value : float, optional
        The value of the chosen response given the stimulus on this trial before reward was or was not delivered.
    obtained_reward : float (0.0 or 1.0), optional
        Indicates whether reward was received at this trial. Will be used to calculate the prediction error (PE) (= reward - previous_value)
    LR : float, optional
        The learning rate (LR) is a scaling factor, which defines the scale with which the PE will be used to update the value of the chosen response.

    Returns
    -------
    PE : float
        The discrepancy between the expected_reward (previous_value) and the obtained reward.
    updated_value : float
        The new value for the chosen response given this stimulus, after updating the value according to the delta-learning rule.

    Description
    -----------
    Function to update the value of the stimulus-response pair based on the current value of this stimulus-response pair and the reward obtained on this trial.
    The value is updated using the delta-learning rule:
        Q(s,a) at time t+1 = Q(s,a) at time t + LR * PE at time t (with PE = reward at time t - Q(s,a) at time t)."""
    #Calculate the prediction error:
    PE = obtained_reward - previous_value  #PE = R(t-1) - V(s, a)(t-1)
    # calculate the new value for this stimulus-response pair
    updated_value = np.sum([previous_value, np.multiply(PE, LR)]) # V(s, a)t = V(s, a)(t-1) + PE*LR
    return PE, updated_value

def simulate_responses_RL(simulation_LR = 0.5, simulation_inverseTemp = 1, design = None):
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

    # responses: array that will contain the responses on each trial
    responses = np.array([])
    ntrials = design.shape[0]
    # values: array containing four values, one for each stimulus-response pair!
        # [[stimulus0 & response0, stimulus0 & response1], [stimulus1 & response0, stimulus1 & response1]]
    values = np.array([[0.5, 0.5], [0.5, 0.5]])
    ##alternatively: np.ones(nstim, nresp)*0.5

    # trial-loop: generate a response on each trial sequentially
    for trial in range(ntrials):

    #Define the variables you'll need for this trial (take them from the design)

        stimulus = np.int(design[trial, 1]) # the stimulus that appears this trial
        CorResp =  np.int(design[trial, 3]) # the correct response on this trial (depends on the stimulus-response mapping rule on this trial)
        FBcon = design[trial, 4] # feedback congruence on this trial: FB is congruent with the current stimulus-response mapping rule if FBCon = 1


    #Simulate the response given by the hypothetical participant. Depending on the value for each response and the inverse_temperature parameter.

        # define which weights are of importance on this trial: depends on which stimulus "appears"
        stimulus_weights = values[stimulus, :]
        # compute probability of each action on this trial (using the weights for each action with the stimulus of this trial)
        response_probabilities = softmax(values = stimulus_weights, inverse_temperature = simulation_inverseTemp)
        # define which action is actually chosen (based on the probabilities)
        response = choose_response(response_probabilities = response_probabilities)
        # store the response given on this trial
        responses = np.append(responses, response)
        #Update the value of the response-action pair that was relevant this trial (stimulus shown & response given)
        #define whether reward was received this trial or not: reward present when:
            # a) response = correct response and FB congruence = 1 (congruent)
            # b) response != correct response and FB congruence = 0 (incongruent)
        reward_present = ((response == CorResp and FBcon == 1) or (response != CorResp and FBcon == 0))*1
        reward_present = np.float(reward_present)
        #compute the PE and the updated value for this trial (and this stimulus-response pair)
            # to compute the PE & updated value, just work with whether reward was present or not
        PE, updated_value = delta_rule(previous_value=values[stimulus, response],
                                            obtained_reward=reward_present, LR=simulation_LR)
        #update the value of the stimulus-response pair that was used this trial
        values[stimulus, response] = updated_value
    return responses

def likelihood_RL(parameter_set, data):
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

#Prepare the likelihood estimation process: make sure all relevant variables are defined
    # the start values for each stimulus-response pair: these are the same as in the simulate_responses function
    values = np.array([[0.5, 0.5], [0.5, 0.5]])
    ntrials = data.shape[0] # deduce number of trials within the experiment from the length of the design matrix
    # Define the response accuracy on each trial: (responses == correct_responses)*1
        # remember: correct_responses depends on the stimulus-response mapping rule and does not necessarily mean reward is delivered
        # 0 = incorrect response, 1 = correct response
    Accuracy = (data[:, 2] == data[:, 3])*1
    # Define whether reward was received each trial: (accuracy = FBcongruence)*1
        # if FBCon = 1: reward is received on trials where a correct response was given
        # if FBCon = 0: reward is received on trials where an incorrect resposne was given
        # 0 = no reward received, 1 = reward received
    actual_rewards = (Accuracy == data[:, 4])*1
    actual_responses = data[:, 2] # the participants' responses (obtained with the simulate_responses function)
    stimuli = data[:, 1] # the stimuli that were shown each trial

#Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0 # this is calculated by summing over trials the log( L(parameter set|response on that trial) )

    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
    #Define the variables that are important for the likelihood estimation process
        stimulus = int(stimuli[trial]) #the stimulus shown on this trial
        response = actual_responses[trial].astype(int) #the response given on this trial
        reward_this_trial = actual_rewards[trial] #the reward presence or absence on this trial

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
        else: loglikelihoods = stimulus_weights*retransformed_invT - np.log(np.sum((np.exp(stimulus_weights[0]*retransformed_invT)+np.exp(stimulus_weights[1]*retransformed_invT))))

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


#%%% Functions used in the PowerAnalysis script
#%%% check parameters
def check_input_parameters_RL(ntrials, nreversals, reward_probability, full_speed, criterion, output_folder):
    variables_are_fine = 1
    if ntrials < 5:
        print("ntrials = {}; but minimal number of trials = 5.".format(ntrials))
        variables_are_fine = 0
    if nreversals >= ntrials:
        print('nreversals = {} and ntrials = {}; but nreversals should be < ntrials.'.format(nreversals, ntrials))
        variables_are_fine = 0
    if reward_probability < 0 or reward_probability > 1:
        print("reward_probability = {}; but should be element of [0, 1].".format(reward_probability))
    if full_speed != 0 and full_speed != 1:
        print("full_speed = {}, but should be either 0 or 1".format(full_speed))
        variables_are_fine = 0
    if criterion !=  'IC' and criterion != 'EC' and criterion !='GD':
        print("criterion = {}, but should be IC, EC or GD".format(criterion))
        variables_are_fine = 0
    if type(output_folder) != str:
        print("output_folder does not exist")
        variables_are_fine = 0
    elif not os.path.isdir(output_folder):
        print('output_folder does not exist')
        variables_are_fine = 0

    return variables_are_fine


def create_design_RL(ntrials = 480, nreversals = 1, reward_probability = 0.8):
    """
    Parameters
    ----------
    ntrials : integer, optional
        The number of trials that will be used in the experiment. The default is 480.
    nreversals : integer, optionalS
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
        The default is 1.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.

    Returns
    -------
    design : numpy array, shape = (ntrials X 5)
        Array defining the relevant variables on each trial.
        Column 0 = Stimulus-response mapping rule for each trial (0 or 1: rule0 or rule1)
        Column 1 = Appearing stimulus on each trial (0 or 1 : stimulus0 or stimulus1)
        Column 2 = Empty column, will be used to fill in which response was given each trial AFTER data generation
        Column 3 = Correct response on each trial: depends on the stimulus and the rule (0 or 1 : response0 or response1)
        Column 4 = Feedback congruence on each trial: whether feedback (reward or not) is congruent with the current stimulus-response mapping rule.
                    If not, one is rewarded for an incorrect response. (0 or 1 : incongruent or congruent)

    Description
    -----------
    Function to create the design that will be used for (1) data simulation and (2) parameter estimation.
    For both these processes the stimulus, correct response and feedback congruence on each trial have to be defined.
    Since the correct response depends on the stimulus-response mapping rule, this rule has to be defined as well.
        Stimulus-response mapping rule: the rule that the participant has to learn in order to maximize reward
            - rule 0: stimulus0 = response1, stimulus1 = response0; rule 1: stimulus0 = response0, stimulus1 = response1
            - the number of rule reversals are specified in the create_design function.
        Stimulus: the stimulus shown each trial
            - in 50% of the trials stimulus0 appears, in 50% of the trials stimulus1
        Response: the response given on each trial
            - empty column of size ntrials, will be filled in after data simulation
        Correct response: the correct response on each trial
            - depends on the stimulus-response mapping rule for this trial
        Feedback congruence: whether feedback is congruent or incongruent with the current stimulus-response action mapping rule
            - depends on the reward_probability defined by the used: if reward_probability = 0.80, then the feedback will be congruent in 80% of the trials

    """
# Create the template for the design: a pandas dataframe of shape (ntrials X 5)

    design_df = pd.DataFrame(index = range(ntrials),
                             columns = ['rule', 'stimulus', 'response', 'CorResp', 'FBCon'], dtype = int)
# Column 0: define the stimulus-response mapping rule for each trial

    # complicated functions: with ntrials%(nreversals+1) !=0, we want the number of trials before a reversal to be roughly the same (with max. 1 trial difference)
    nchanges = nreversals+1
    nrule_repetitions = int(ntrials/nchanges)
    rest = ntrials%nchanges
    x1 = np.tile(np.repeat([0, 1], nrule_repetitions), int(np.ceil((nchanges-rest)/2)))[:nrule_repetitions*(nchanges-rest)]
    if x1[-1] == 1: x2 = np.tile(np.repeat([0, 1], nrule_repetitions+1), int(np.ceil(rest/2)))[:(nrule_repetitions+1)*rest]
    else: x2 = np.tile(np.repeat([1, 0], nrule_repetitions+1), int(np.ceil(rest/2)))[:(nrule_repetitions+1)*rest]
    design_df['rule'] = np.concatenate([x1, x2])

# Column 1: define which stimulus is shown each trial

    # each stimulus appears in 50% of the trials, in a random sequence
    stimuli = np.concatenate([np.zeros(int(np.floor(ntrials/2))), np.ones(int(np.ceil(ntrials/2)))]).astype(int)
    np.random.shuffle(stimuli)
    design_df['stimulus'] = stimuli

# Column 3: define the correct response at each trial

    #correct response depends on the stimulus-response mapping rule
    for trial in range(design_df.shape[0]):
        if design_df.loc[trial, 'rule'] == 1: design_df.loc[trial, 'CorResp'] = design_df.loc[trial, 'stimulus']
        else: design_df.loc[trial, 'CorResp'] = (not design_df.loc[trial, 'stimulus'])*1

# Column 4: define the feedback congruence at each trial

    # feedback is congruent in i% of the trials with i = reward_probability*100
    ncongruent = int(np.round(ntrials*reward_probability, 0))
    FBcon_array = np.concatenate([np.ones(ncongruent), np.zeros(ntrials-ncongruent)])
    np.random.shuffle(FBcon_array)
    design_df['FBCon'] = FBcon_array

# Convert the design dataframe to a design array, since arrays are computationally less demanding

    design = design_df.to_numpy(dtype = int)
    return design

def Incorrelation_repetition_RL(inverseTemp_distribution, LR_distribution, npp, ntrials, start_design, rep, nreps, ncpu):
    """

    Parameters
    ----------
    inverseTemp_distribution : numpy array, shape = (2,)
        Defines the mean & standard deviation of the normal distribution that will be used to draw the true inverse Temperature values from for each hypothetical participant.
        Mean of the distribution = inverseTemp_distribution[0], standard deviation of the distribution = inverseTemp_distribution[1]
    LR_distribution : numpy array, shape = (2,)
        Defines the mean & standard deviation of the normal distribution that will be used to draw the true learning rate values from for each hypothetical participant.
        Mean of the distribution = LR_distribution[0], standard deviation of the distribution = LR_distribution[1].
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
    True_LRs =  generate_parameters(mean = LR_distribution[0], std = LR_distribution[1], npp = npp)
    True_inverseTemps = generate_parameters(mean = inverseTemp_distribution[0], std = inverseTemp_distribution[1], npp = npp)

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
    Statistic = np.round(np.corrcoef(True_LRs, LRestimations)[0, 1], 2)

    if rep == 0:
        t1 = time.time() - t0
        estimated_seconds = t1 * np.ceil(nreps / ncpu)
        estimated_time = np.ceil(estimated_seconds / 60)
        print("\nThe power analysis will take ca. {} minutes".format(estimated_time))
    # return proportion_failed_estimates, Statistic
    return Statistic
