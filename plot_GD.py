import os,sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ssms
# sys.path.append(r"D:\horiz\IMPORTANT\0study_graduate\Pro_COMPASS\COMPASS_DDM\results\test3")
plot_heatmap_GD = 1
c=1


ResultPath = "results\\test6_GD"
DDM_id = "ddm"

s_pooled = 0.1 # see filename
cohens_d = 0.5 # choen's d

par_ind = [2]
nreps = 20

range_ntrials = [20,40,60]
range_npp = [20,40,60]

p_list = []
for p in range(len(par_ind)):
    p_list.append(ssms.config.model_config[DDM_id]["params"][par_ind[p]])

heatmap_p = np.zeros((len(range_ntrials),len(range_npp)))
heatmap_p_c = np.zeros((len(range_ntrials),len(range_npp)))

for p in range(len(par_ind)):
    for n_t in range(len(range_ntrials)):
        for n_p in range(len(range_npp)):
            ntrials = range_ntrials[n_t]
            npp_pergroup = range_npp[n_p]

            
            OutputFile_name = 'OutputGD{}P{}SD{}T{}N{}M{}ES.csv'.format(par_ind[p],np.round(s_pooled,2),ntrials, 
                                                                        npp_pergroup, nreps,np.round(cohens_d,2))
            OutputFile_path = os.path.join(os.getcwd(), ResultPath, OutputFile_name)


            OutputResults = pd.read_csv(OutputFile_path, delimiter = ',')

            PowerFile_name = 'PowerGD{}P{}SD{}T{}N{}M{}ES.csv'.format(par_ind[p],np.round(s_pooled,2),ntrials, 
                                                                        npp_pergroup, nreps,np.round(cohens_d,2))
            PowerFile_path = os.path.join(os.getcwd(), ResultPath, PowerFile_name)


            PowerResults = pd.read_csv(PowerFile_path, delimiter = ',')[p_list[p]].dropna(axis = 0)
            conven_cor = pd.read_csv(PowerFile_path, delimiter = ',')['conventional_power'].dropna(axis = 0)

            heatmap_p[n_t,n_p] = PowerResults
            heatmap_p_c[n_t,n_p] = conven_cor
            
Power_AllData = (heatmap_p,)
conven_AllData = (heatmap_p_c,)

if plot_heatmap_GD:
    fontsize = 20
    xylabels = ['participants','trials']
    xticklabels = range_npp
    yticklabels = range_ntrials
    y_visible = 1
    colbar = 1
    fig, axs = plt.subplots(1,len(p_list))
                            # gridspec_kw={
                            #     'width_ratios': [1, 1,1,1.25]
                            #     # 'height_ratios': [1, 1,1,1]
                            #     })

    fig.suptitle("Group difference(cohen's d) = {} with Nreps = {}".format(cohens_d,nreps), fontsize=fontsize)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sns.set(font_scale=1.4)

    def plot_MultiHeatmap(ax_fi,fi,fi_range,data,norm,title,xticklabels,yticklabels,xylabels, t = 0.5):
        y_visible = False
        colbar = False
        if fi == fi_range[0]:
            y_visible = True
        if fi == fi_range[1]:
            colbar = True
        
        ax=sns.heatmap(data,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        annot=True,
                        ax = ax_fi,
                        cbar=colbar, 
                        cmap = 'Blues',
                        norm=norm
                    )
        
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(xylabels[0])  # x轴标题
        ax.set_ylabel(xylabels[1])
        ax.axes.yaxis.set_visible(y_visible)
        ax.invert_yaxis()
        figure = ax.get_figure()
        return figure

    for i_p in range(len(p_list)):


        data=pd.DataFrame(Power_AllData[i_p])
        c_data = pd.DataFrame(conven_AllData[i_p])

        ax=sns.heatmap(data,
                            xticklabels=xticklabels,
                            yticklabels=yticklabels,
                            annot=True,
                            cbar=colbar, 
                            cmap = 'Blues',
                            norm=norm
                        )
            
        ax.set_title("power of parameter {}".format(p_list[i_p]), fontsize=18)
        ax.set_xlabel(xylabels[0])  # x轴标题
        ax.set_ylabel(xylabels[1])
        ax.axes.yaxis.set_visible(y_visible)
        ax.invert_yaxis()
        figure = ax.get_figure()
        plt.show()
        
        ax=sns.heatmap(c_data,
                            xticklabels=xticklabels,
                            yticklabels=yticklabels,
                            annot=True,
                            cbar=colbar, 
                            cmap = 'Blues',
                            norm=norm
                        )
            
        ax.set_title("conventional power of parameter {}".format(p_list[i_p]), fontsize=18)
        ax.set_xlabel(xylabels[0])  # x轴标题
        ax.set_ylabel(xylabels[1])
        ax.axes.yaxis.set_visible(y_visible)
        ax.invert_yaxis()
        figure = ax.get_figure()

        plt.show()

    plt.show()
    a =0