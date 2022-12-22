# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:49:29 2022

@author: maudb
"""

import pandas as pd 
import os, itertools
import numpy as np

def Create_InputFileIC(ntrials, ireversal, npp, meanLR, sdLR, meanInvT, sdInvT, Prew, tau, output_folder, inputfile_folder): 
    
    path = inputfile_folder
    file = os.path.join(path, "InputFile_IC.csv")
    columns = pd.read_csv(file, sep = ';').columns
    df = pd.DataFrame(columns = columns)
    
    trials_ppcombo = np.array(list(itertools.product(ntrials, npp)))
    
    df['ntrials'] = trials_ppcombo[:, 0]
    df['npp'] = trials_ppcombo[:, 1]
    df['nreversals'] = np.array(trials_ppcombo[:, 0]/ireversal-1, dtype= int)
    df['meanLR'] = meanLR
    df['sdLR'] = sdLR
    df['meanInverseTemperature'] = meanInvT
    df['sdInverseTemperature'] = sdInvT
    df['reward_probability'] = Prew
    df['tau']= tau
    df['output_folder'] = output_folder
    
    
    inputfile_simulations = os.path.join(inputfile_folder, "InputFile_IC_simulations.csv")
    if not os.path.isfile(inputfile_simulations): df.to_csv(inputfile_simulations, index = False, sep = ';')
    else: print("Warning: filename already exists")

def Create_InputFileGD(ntrials, ireversal, npp, meanLRs, sdLRs, meanInvTs, sdInvTs, Prew, typeIerror, output_folder, inputfile_folder): 
    path = inputfile_folder
    file = os.path.join(path, "InputFile_GD.csv")
    columns = pd.read_csv(file, sep = ';').columns
    df = pd.DataFrame(columns = columns)
    
    trials_ppcombo = np.array(list(itertools.product(ntrials, npp)))
    
    df['ntrials'] = trials_ppcombo[:, 0]
    df['npp_group'] = np.array(trials_ppcombo[:, 1]/2, dtype = int)
    df['nreversals'] = np.array(trials_ppcombo[:, 0]/ireversal-1, dtype = int)
    df['meanLR_g1'], df['meanLR_g2'] = meanLRs[0], meanLRs[1]
    df['sdLR_g1'], df['sdLR_g2'] = sdLRs[0], sdLRs[1]
    df['meanInverseTemperature_g1'], df['meanInverseTemperature_g2'] = meanInvTs[0], meanInvTs[1] 
    df['sdInverseTemperature_g1'], df['sdInverseTemperature_g2'] = sdInvTs[0], sdInvTs[1] 
    df['reward_probability'] = Prew
    df['TypeIerror']= typeIerror
    df['output_folder'] = output_folder
    
    inputfile_simulations = os.path.join(inputfile_folder, "InputFile_GD_simulations.csv")
    if not os.path.isfile(inputfile_simulations): df.to_csv(inputfile_simulations, index = False, sep = ';')
    else: print("Warning: filename already exists")
    

df = Create_InputFileIC(ntrials = np.arange(80, 1000, 160), ireversal = 40, npp = np.arange(40, 201, 20), 
                   meanLR = 0.7, sdLR = 0.1, meanInvT = 1.5, sdInvT = 0.5, Prew = 0.8, tau = 0.75, 
                   output_folder = r'/data/gent/442/vsc44254/COMPASS/Version2022_2023', inputfile_folder = r'C:\Users\maudb\Documents\GitHub\COMPASS')    

df = Create_InputFileGD(ntrials = np.arange(80, 1000, 160), ireversal = 40, npp = np.arange(40, 201, 20), 
                   meanLRs = [0.675, 0.725], sdLRs = [0.1, 0.1], meanInvTs = [1.5, 1.5], sdInvTs = [0.5, 0.5], Prew = 0.8, 
                   typeIerror = 0.05, 
                   output_folder = r'/data/gent/442/vsc44254/COMPASS/Version2022_2023', inputfile_folder = r'C:\Users\maudb\Documents\GitHub\COMPASS')    


    






