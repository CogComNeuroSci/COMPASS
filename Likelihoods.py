# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 23:03:42 2023

@author: horiz
"""

import math
from wfpt import wiener_like
from wfpt_n.wfpt_n import wiener_like_n
import numpy as np

def neg_likelihood(param,arg):
    err = 10 ** (-10)
    data = arg[0]
    DDM_id = arg[1]
    if DDM_id =="ddm":
        llh = wiener_like_n(data,param[0],0,
                                param[1],
                                param[2],0,
                                param[3],0,err)
# =============================================================================
#         parameters of wienr_like
#         def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
#                         double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
#                         double p_outlier=0, double w_outlier=0.1):     
# =============================================================================

    return -llh