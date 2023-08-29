import os,sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ssms
sys.path.append(r"D:\horiz\IMPORTANT\0study_graduate\Pro_COMPASS\COMPASS_DDM\results\test")
ResultPath = "results\\test"
ntrials = 80
npp = 80
nreps = 30
DDM_id = "ddm"


Parameters = ssms.config.model_config[DDM_id]["params"]
OutputFile_name = 'OutputIC{}T{}N{}M.csv'.format(ntrials,npp, nreps)
OutputFile_path = os.path.join(os.getcwd(), ResultPath, OutputFile_name)
OutputResults = pd.read_csv(OutputFile_path, delimiter = ',')

PowerFile_name = 'PowerIC{}T{}N{}M.csv'.format(ntrials,npp, nreps)
PowerFile_path = os.path.join(os.getcwd(), ResultPath, PowerFile_name)
PowerResults = pd.read_csv(PowerFile_path, delimiter = ',')[Parameters].dropna(axis = 0)




# tau = OutputResults['tau'].values[-1]
tau = 0.5
for p in Parameters :
    
    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    sns.set_theme(style = "white",font_scale=1.4)
    sns.kdeplot(OutputResults[p].dropna(axis = 0), ax = axes)
    fig.suptitle("Pr(Correlation >= {}) with {} pp, {} trials)".format(tau, npp, ntrials), fontweight = 'bold')

    power_estimate = PowerResults[p].values
    axes.set_title("Power = {} based on {} reps".format(np.round(power_estimate*100, 2), nreps))
    axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
    axes.tick_params(labelsize=20)
    axes.set_xlabel('Correlations of '+p,fontsize = 20)
    axes.set_ylabel('Density',fontsize = 20)

    plt.show()
    # axes.set_xticklabels(fontsize=14)
    # axes.set_yticklabels(fontsize=14)


                #     fig, axes = plt.subplots(nrows = 1, ncols = 1)
                # sns.kdeplot(output["Statistic"], label = "T-statistic", ax = axes)
                # fig.suptitle("Pr(T-statistic > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold')
                # axes.set_title("Power = {}% \nbased on {} reps with Cohen's d = {}".format(np.round(power_estimate*100, 2), nreps, np.round(cohens_d,2)))
                # axes.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
