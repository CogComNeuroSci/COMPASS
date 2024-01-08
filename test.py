
import hddm_wfpt
import numpy as np
from numpy.random import rand, seed, uniform
from datetime import datetime
from Functions_DDM import *
import seaborn as sns
import math
import matplotlib
import matplotlib.pyplot as plt


import pyximport
pyximport.install()

from wfpt_n.wfpt_n import wiener_like_n

parameter_range = 0
parameter_recovery = 1

test_old_wiener_like = 0
test_likelihood_function = 0


random_parameters = 0
EC = 0
GD = 0
test_gen = 0


if random_parameters :
    # seed(0)
    #%%
    v = uniform(-3,3)
    
    a = uniform(0.3,2.5)
    z = uniform(0.1,0.9)
    t = uniform(0,2)
    

    z_nonorm = a * z
    rt = rand() * 4 + t
    err = 10 ** (round(rand() * -10))
else: 
    v = 0.6
    a = 1
    z = 0
    t = 0.5
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
    plt.show()

def plot_ParameterRecovery(param_bounds,name,Est,Tru,fig_size = [10,10],xylabelsize=20,labelsize=20,titlesize=28):
    # if len(ACC) ==1:
    #     ACC = np.ones((len(Est[name]),1))

    min_lim = param_bounds[0]
    max_lim = param_bounds[1]
    fig = plt.figure()
    fig.set_size_inches(fig_size[0],fig_size[1])
    plt.scatter(Est[name],y=Tru[name]) 
    # plt.scatter(Est[name],y=Tru[name],alpha=ACC)
    plt.xlim(min_lim,max_lim)
    plt.ylim(min_lim,max_lim)
    l = np.arange(min_lim,max_lim,(max_lim-min_lim)/200)
    plt.plot(l,l)
    plt.title('Parameter Recovery of '+name,fontdict={"fontsize" : titlesize})
    plt.xlabel('estimated',fontdict={"fontsize" : xylabelsize})
    plt.ylabel('true',fontdict={"fontsize" : xylabelsize}) 


    plt.tick_params(labelsize=labelsize)

    plt.show()

def plot_Heatmap(ax_fi,fi,fi_range,data,title,norm,xticklabels,yticklabels,xylabels, t = 0.5):
    

    y_visible = False
    colbar = False
    if fi == fi_range[0]:
        y_visible = True
    if fi == fi_range[1]:
        colbar = True
    
    ax=sns.heatmap(data,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    annot=False,
                    ax = ax_fi,
                    cbar=colbar, 
                    norm=norm
                )
    
    ax.set_title(title)
    ax.set_xlabel(xylabels[0])  # x轴标题
    ax.set_ylabel(xylabels[1])
    ax.axes.yaxis.set_visible(y_visible)
    
    figure = ax.get_figure()
    return figure

### uniform x & single llh of full_pdf
# full_pdf_wfpt=np.array(np.zeros(200))
# for i in range(200):
#     full_pdf_wfpt[i] = hddm_wfpt.wfpt.full_pdf(np.array(x[i]).reshape(1), v, 0, a, z, 0, t, 0, err, 0)
#     # print(np.around((full_pdf_wfpt[i]),3))

# ImPlot_plot('Likelihood(full-pdf function) on RT ',
#             x,'RT',
#             full_pdf_wfpt,"Likelihood")

# plt.hist(x,bins = 30)


### test wiener_like


if test_old_wiener_like :
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
if test_likelihood_function :
    neg_LLH_WL_wfpt = np.array(np.zeros(200))
    LLH_exp_WL_wfpt = np.array(np.zeros(200))
    for i in range(200):
        arg = (np.array(x[i]).reshape(1),DDM_id)
        neg_LLH_WL_wfpt[i] = neg_likelihood(params,arg)
        LLH_exp_WL_wfpt [i]= math.exp(-neg_LLH_WL_wfpt[i])

    print(params)
    ImPlot_plot('Likelihood(likelihood function) on RT ',
                x,'RT',
                LLH_exp_WL_wfpt,"Likelihood_LF")

    ImPlot_plot('Neg_log_likelihood(likelihood function) on RT ',
                x,'RT',
                neg_LLH_WL_wfpt,"-log(f)")

### parameter recovery
if parameter_recovery:
    
    npp = 50
    nreps = 10
    ntrials = 50
    estimation_results = np.zeros((50,4))

    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])

    start_time = datetime.now()
    
    for rep in range(nreps):
        r,Tru,Est,ACC,RT = Incorrelation_repetition_DDM(means = np.array([0,1.18,0.5,0.5]),stds = np.array([0.3,0.33,0.05,0.25]), 
                                    param_bounds = param_bounds, 
                                    npp = npp,method = "Nelder-Mead",
                                    ntrials = npp, DDM_id = "ddm", rep=rep, nreps = 50, ncpu = 6)

        print("npp = {}, ntrails = {}".format(npp,ntrials))

        #%% plot`
        sns.set_theme(style = "white",font_scale=1.4)
        sns.kdeplot(Tru["v"].dropna(axis = 0))

        plot_ParameterRecovery([-0.8,0.8],"v",Est,Tru)
        plot_ParameterRecovery([0.3,2.5],"a",Est,Tru)
        plot_ParameterRecovery([0.3,0.7],"z",Est,Tru)
        plot_ParameterRecovery([0,1.5],"t",Est,Tru)

    # plt.figure()
    # plt.scatter(Est[:,1],y=Tru['a'])
    # plt.xlim(param_bounds[0,1],param_bounds[1,1])
    # plt.ylim(param_bounds[0,1],param_bounds[1,1])
    # l = np.arange(0.3,2.5,2.1/200)
    # plt.plot(l,l)
    # plt.title('Parameter Recovery of a')
    # plt.xlabel('estimated')
    # plt.ylabel('true')
    # plt.show

    # plt.figure()
    # range = 0.1,0.9   
    # plt.scatter(Est[:,2],y=Tru['z'])
    # plt.xlim(range)
    # plt.ylim(range)
    # l = np.arange(0.1,0.9,0.8/200)
    # plt.plot(l,l)
    # plt.title('Parameter Recovery of z')
    # plt.xlabel('estimated')
    # plt.ylabel('true')
    # plt.show
    
    # plt.figure()
    # range = 0,2
    # plt.scatter(Est[:,3],y=Tru['t'])
    # plt.xlim(range)
    # plt.ylim(range)
    # l = np.arange(0,2,2/200)
    # plt.plot(l,l)
    # plt.title('Parameter Recovery of t')
    # plt.xlabel('estimated')
    # plt.ylabel('true')
    # plt.show


if parameter_range:
    # paremeters
    # reponse
    # ACC       
    # 
    n_bin = 20
    ntrials = 50

    bounds = ssms.config.model_config[DDM_id]['param_bounds']
    v = np.around(list(np.arange(-3.01,3.01,6/n_bin)) ,2)
    a = np.around(list(np.arange(0.3,2.5,2.2/n_bin)) ,2)
    z = [0.2,0.4,0.6,0.8]
    t = 0.5

    max_RT = 8


    ACC_out = np.empty((n_bin,n_bin))
    RT_out = np.empty((n_bin,n_bin))


    fig, axs = plt.subplots(1,len(z), 
                       gridspec_kw={
                           'width_ratios': [1, 1,1,1.25]
                           # 'height_ratios': [1, 1,1,1]
                           })
    fig.suptitle("Heapmap for RT", fontsize=30)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_RT)

    for i_z in range(len(z)):
        for i_v in range(n_bin):
            for i_a in range(n_bin):

                responses = simulate_responses(np.array([v[i_v],a[i_a],z[i_z],t]),DDM_id,ntrials)
                responses = np.array(responses['rts'] * responses['choices'])

            # validation of parameters

                ACC = np.mean( responses*v[i_v] > 0)
                # if ACC <= 0.5 or ACC >= 0.95:
                    # ACC = -0.1
                RT = np.mean(np.abs(responses))
                if RT <= t or ACC >= max_RT:
                    RT = max_RT


                ACC_out[i_v,i_a] = ACC
                RT_out[i_v,i_a] = RT

    
                
        data=pd.DataFrame(RT_out)
        f_z = plot_Heatmap(axs[i_z],i_z,[0,len(z)-1],data,title = "z = {}, t = {}".format(z[i_z],t),
                           norm= norm,
                        xticklabels = a,yticklabels = v,
                        xylabels = ['a','v'], t = 0.5)
    
        
    
    
    plt.show()
    save = 0
    if save:
        f_z.savefig('D:/horiz/IMPORTANT/0study_graduate/Pro_COMPASS/COMPASS_DDM/results/test3/ParRange_0.4z0.5t.png')

if EC :


    npp = 50
    ntrials = 50
    estimation_results = np.zeros((50,4))

    param_bounds = np.array(ssms.config.model_config[DDM_id]['param_bounds'])

    Esti_r, Esti_pValue, True_r, True_pValue = Excorrelation_repetition_DDM(means = np.array([0,1.18,0.5,0.5]),stds = np.array([0.3,0.33,0.05,0.25]), 
                                param_bounds = param_bounds, 
                                par_ind = 2, DDM_id = 'ddm',true_correlation = 0.6, npp = npp, ntrials = ntrials, rep = 1, nreps = 50, ncpu = 1)
    
if GD:
    means_g1 = [0,1.4,0.5,1]
    means_g2 = [0,1.0,0.5,1]
    stds_g1 =  [2,0.7,0.3,0.3]
    stds_g2 =  [2,0.7,0.3,0.3]
    par_ind = 1
    param_bounds= np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    statistic, p = Groupdifference_repetition_DDM(means_g1, stds_g1,means_g2, stds_g2,DDM_id, par_ind , param_bounds,
                                   npp_per_group = 20, ntrials = 20, rep =1 , nreps = 10, ncpu = 6, standard_power = False)
    
    

if test_gen:
    param_bounds= np.array(ssms.config.model_config[DDM_id]['param_bounds'])
    p = generate_parameters_DDM(means = [0,1.18,0.5,0.5],stds = [1,1,0.1,0.25], 
                                param_bounds= np.array(ssms.config.model_config[DDM_id]['param_bounds']) , par_ind = 0 , 
                        npp = 400, multivariate = False, corr = False)