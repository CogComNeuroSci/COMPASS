# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:46:54 2023

@author: horiz
"""

from scipy import optimize
import numpy as np

def MLE(fun,arg,param_bounds,method, show = 0):
    ranges = ()
    for p in range(param_bounds.shape[1]) :
        ranges = ranges+(tuple(param_bounds.T[p]),)
        
    num_par = param_bounds.shape[1]    
    if method=="Brute": 
        # grid research method
        # define rangs of parameters
        ranges = ()
        for p in range(param_bounds.shape[1]) :
            ranges = ranges+(tuple(param_bounds.T[p]),)
        estimated_parameters = optimize.brute(fun, ranges, args = (arg,) ,finish=None) 

    elif method=="Nelder-Mead":
        estimated_parameters_MulIni = np.empty((10 * param_bounds.shape[1],param_bounds.shape[1]+1))
        for r in range(10 * num_par): # loop for multiple initial guesses
            
            start_params = np.random.uniform(param_bounds[0],param_bounds[1])
            if show :
                print("initial guess:",r)
                print('start point',start_params,fun(start_params,arg))
            # start_params = True_Par.loc[pp] + np.random.rand()
            
            # llh_test = fun(start_params,responses,DDM_id)
            optimization_output = optimize.minimize(fun, start_params,args = (arg,),
                                                    method="Nelder-Mead",
                                                    bounds= ranges,
                                                    # bounds= tuple(param_bounds.T.reshape(num_par,2)),
                                                    options = {'maxfev':1000, 'xatol':0.01, 'return_all':1})
            estimated_parameters_SinIni = optimization_output['x']
            if show :
                print('end point',estimated_parameters_SinIni,fun(estimated_parameters_SinIni,arg)) 
            estimated_parameters_MulIni[r,0:num_par] = estimated_parameters_SinIni
            estimated_parameters_MulIni[r,num_par] = fun(estimated_parameters_SinIni,arg)

        estimated_parameters = estimated_parameters_MulIni[np.argmin(list(estimated_parameters_MulIni[:,-1])),0:num_par]

    return estimated_parameters
                                                           