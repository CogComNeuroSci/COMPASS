# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:27:56 2022

@author: maudb
"""

import os, sys
import pandas as pd
import numpy as np

def create_folders(criterion = 'IC', inputfile_folder = os.getcwd()):
    InputFile_name = "InputFile_{}_simulations.csv".format(criterion)
    InputFile_path = os.path.join(inputfile_folder, InputFile_name)
    InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
    InputDictionary = InputParameters.to_dict()
    print(InputDictionary)

    for row in range(InputParameters.shape[0]):
        ntrials = InputDictionary['ntrials'][row]
        nreversals = InputDictionary['nreversals'][row]
        output_folder = InputDictionary['output_folder'][row]
        output_dir = os.path.join(output_folder, 'Output')

        if criterion == 'IC':
            npp = InputDictionary['npp'][row]
            npp = InputDictionary['npp'][row]
            s_pooled = InputDictionary['sdLR'][row]
            ES_text = ''

        elif criterion == 'GD':
            npp_pergroup = InputDictionary['npp_group'][row]
            npp = npp_pergroup*2
            meanLR_g1, sdLR_g1 = InputDictionary['meanLR_g1'][row], InputDictionary['sdLR_g1'][row]
            meanLR_g2, sdLR_g2 = InputDictionary['meanLR_g2'][row], InputDictionary['sdLR_g2'][row]

            s_pooled = np.sqrt((sdLR_g1**2 + sdLR_g2**2) / 2)
            cohens_d = np.abs(meanLR_g1-meanLR_g2)/s_pooled
            ES_text = '{}ES'.format(np.round(cohens_d, 1))

        elif criterion == 'EC':
            npp = InputDictionary['npp'][row]
            s_pooled = InputDictionary['sdLR'][row]
            True_correlation = InputDictionary['True_correlation'][row]

            ES_text = '{}ES'.format(np.round(True_correlation, 1))


        Results_folder = os.path.join(output_dir, "Results{}{}SD{}{}T{}R{}N".format(criterion, s_pooled,
                                                                                    ES_text,
                                                                                    ntrials,
                                                                                    nreversals, npp))
        if not os.path.isdir(Results_folder): os.makedirs(Results_folder)

input_parameters = sys.argv[1:]
assert len(input_parameters) == 1
criterion = input_parameters[0]

create_folders(criterion = criterion, inputfile_folder = os.getcwd())
