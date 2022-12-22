# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:59:30 2022

@author: maudb
"""

"""Script to combine the statistic outputs per N & T combo into a single file
- input: $Combine_Statistics.py 'criterion' 'SD' 'ES'"""


import os, itertools, sys
import numpy as np

def combine_output(criterion = 'IC', variable = 'Statistic', ntrials = np.array([80]), ireversal = 40,
                   npp = np.array([60]), sd = 0.2, main_folder = os.getcwd(),
                   nreps = 100, ES = 0.8):
    if criterion == 'IC': ES_text = ''
    elif criterion == 'GD': ES_text = '{}ES'.format(ES)
    elif criterion == 'EC': ES_text = '{}ES'.format(ES)
    trials_ppcombo = np.array(list(itertools.product(ntrials, npp)))
    read_folder = os.path.join(main_folder, 'Output')
    for itrials, ipp in zip(trials_ppcombo[:, 0], trials_ppcombo[:, 1]):

        nreversals = int(itrials/ireversal-1)

        folder = os.path.join(read_folder, 'Results{}{}SD{}{}T{}R{}N'.format(criterion, sd,
                                                                             ES_text,
                                                                             itrials, nreversals, ipp))
        Stats = np.array([])
        for irep in range(1, nreps+1):

            Statistic = np.load(os.path.join(folder, "{}_rep{}.npy".format(variable, irep)))
            Stats = np.append(Stats, Statistic)
        np.save(os.path.join(main_folder, 'Stats{}{}SD{}{}T{}R{}N{}reps.npy'.format(criterion, sd,
                                                                                    ES_text, itrials,
                                                                                    nreversals, ipp,
                                                                                    nreps)), Stats)


input_parameters = sys.argv[1:]
assert len(input_parameters) == 4
criterion = input_parameters[0]
SD = input_parameters[1]
ES = input_parameters[2]
nreps = int(input_parameters[3])


combine_output(criterion = criterion, variable = 'Statistic', ntrials = np.arange(80, 1000, 160),
               ireversal = 40, npp = np.arange(40, 201, 20), sd = SD, ES = ES,
               main_folder = r'/data/gent/430/vsc43099/COMPASS',
               nreps = nreps)
