
import hddm_wfpt
import numpy as np
from numpy.random import rand, seed
from datetime import datetime
from Functions_DDM import *
import math
import matplotlib.pyplot as plt

from Likelihoods import neg_likelihood
from ParameterEstimation import MLE

seed(0)
#%%
v = (rand() - 0.5) * 1.5
t = rand() * 0.5
a = 1.5 + rand()
z = 0.5 * rand()
z_nonorm = a * z
rt = rand() * 4 + t
err = 10 ** (round(rand() * -10))
# to see parameters included and its order
# ssms.config.model_config[DDM_id]

DDM_id = "ddm"
nreps = 250



x = list(np.arange(-3,2,5/200))
params = np.array([v, a, z, t])


def ImPlot_plot(title,x,xlabel,y,ylabel,type = "plot",titlesize=35,xylabelsize=25,labelsize = 25,linewidth = 5):
    
    plt.title(title,fontdict={"fontsize" : titlesize})
    if type == "plot":
        plt.plot(x,y,linewidth=linewidth)
    elif type == "scatter":
        plt.scatter(x,y,s=linewidth)
    plt.xlabel(xlabel,fontdict={"fontsize" : xylabelsize})
    plt.tick_params(labelsize=labelsize)
    plt.ylabel(ylabel,fontdict={"fontsize" : xylabelsize})
    plt.show

### uniform x & single llh of full_pdf
full_pdf_wfpt=np.array(np.zeros(200))
for i in range(200):
    full_pdf_wfpt[i] = hddm_wfpt.wfpt.full_pdf(np.array(x[i]).reshape(1), v, 0, a, z, 0, t, 0, err, 0)
    # print(np.around((full_pdf_wfpt[i]),3))

ImPlot_plot('Likelihood(full-pdf function) on RT ',
            x,'RT',
            full_pdf_wfpt,"Likelihood")

plt.hist(x,bins = 30)
### test wiener_like
WL_wfpt = np.array(np.zeros(200))
exp_WL_wfpt = np.array(np.zeros(200))

# python_wfpt_sum = hddm_wfpt.wfpt.wiener_like(responses, v, 0, a, z, 0, t, 0, err, 0)

for i in range(200):
    WL_wfpt[i] = hddm_wfpt.wfpt.wiener_like(np.array(x[i]).reshape(1),
                                     params[0],0,
                                     params[1],
                                     params[2],0,
                                     params[3],0,err)
    exp_WL_wfpt [i]= math.exp(WL_wfpt[i])

ImPlot_plot('Likelihood(Wiener_like function in hddm_wfpt ) on RT ',
            x,'RT',
            exp_WL_wfpt,"likelihood_WL")

ImPlot_plot('Log likelihood(Wiener_like function in hddm_wfpt) on RT ',
            x,'RT',
            WL_wfpt,"Log_likelhood_WL")



### test likelihood function
neg_LLH_WL_wfpt = np.array(np.zeros(200))
LLH_exp_WL_wfpt = np.array(np.zeros(200))
for i in range(200):
    arg = (np.array(x[i]).reshape(1),DDM_id)
    neg_LLH_WL_wfpt[i] = neg_likelihood(params,arg)
    LLH_exp_WL_wfpt [i]= math.exp(-neg_LLH_WL_wfpt[i])

ImPlot_plot('Likelihood(likelihood function) on RT ',
            x,'RT',
            LLH_exp_WL_wfpt,"Likelihood_LF")

ImPlot_plot('Log_likelihood(likelihood function) on RT ',
            x,'RT',
            neg_LLH_WL_wfpt,"-log(f)")

### parameter recovery

estimation_results = np.zeros((50,4))
     

start_time = datetime.now()
print("Power estimation started at {}.".format(start_time))
r,Tru,Est = Incorrelation_repetition(means = np.array([0,1.6,0.5,1]),stds = np.array([2,0.7 ,0.3,0.3]), 
                             param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds']), 
                             npp = 50,method = "Nelder-Mead" ,
                             ntrials = 250, DDM_id = "ddm", rep=1, nreps = 250, ncpu = 6)
ene_time = datetime.now()

#%% plot
if 1 :
        
    plt.figure()
    plt.scatter(Est[:,0],y=Tru['v'])
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    l = np.arange(-3,3,3/200)
    plt.plot(l,l)
    plt.title('Parameter Recovery of v')
    plt.xlabel('estimated')
    plt.ylabel('true')
    plt.show
    

    plt.figure()
    plt.scatter(Est[:,1],y=Tru['a'])
    plt.xlim(0.3,2.5)
    plt.ylim(0.3,2.5)
    l = np.arange(0.3,2.5,2.1/200)
    plt.plot(l,l)
    plt.title('Parameter Recovery of a')
    plt.xlabel('estimated')
    plt.ylabel('true')
    plt.show

    plt.figure()
    range = 0.1,0.9   
    plt.scatter(Est[:,2],y=Tru['z'])
    plt.xlim(range)
    plt.ylim(range)
    l = np.arange(0.1,0.9,0.8/200)
    plt.plot(l,l)
    plt.title('Parameter Recovery of z')
    plt.xlabel('estimated')
    plt.ylabel('true')
    plt.show
    
    plt.figure()
    range = 0,2
    plt.scatter(Est[:,3],y=Tru['t'])
    plt.xlim(range)
    plt.ylim(range)
    l = np.arange(0,2,2/200)
    plt.plot(l,l)
    plt.title('Parameter Recovery of t')
    plt.xlabel('estimated')
    plt.ylabel('true')
    plt.show