import os,sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ssms
# sys.path.append(r"D:\horiz\IMPORTANT\0study_graduate\Pro_COMPASS\COMPASS_DDM\results\test3")
plot_heatmap_GD = 1
c=0
criterion = "IC"

ResultPath = "\\results\Talk\IC\\"
DDM_id = "ddm"

par_ind = [0]
pn = "drift rates"

nreps = 50

range_ntrials = [40,60,80,100,120]
range_npp = [20,40,60,80,100]

s_pooled = 0.3 # see filename
if criterion=="GD":
    ana_list = [0.2, 0.5, 0.8] # choen's d
elif criterion=="EC":
    ana_list = [0.1,0.3,0.5]
elif criterion=="IC":
    ana_list = [0.1,0.2,0.3]


Data_folder = os.getcwd()+ResultPath

# if multiple parameters
p_list = []
for p in range(len(par_ind)):
    p_list.append(ssms.config.model_config[DDM_id]["params"][par_ind[p]])

heatmap_p = np.zeros((len(range_ntrials),len(range_npp)))
heatmap_p_c = np.zeros((len(range_ntrials),len(range_npp)))

def plot_MultiHeatmap(ax_fi,fi,fi_range,data,norm,title,xticklabels,yticklabels,xylabels):
    y_visible = True
    colbar = False

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


    if fi == fi_range-1:
        colbar = True
        plt.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.025, 0.7])  # 这里的数值可能需要根据实际情况进行调整
        fig.colorbar(ax.collections[0], cax=cbar_ax)   
    return ax, figure
fig, axs = plt.subplots(1,len(ana_list))
                        # gridspec_kw={
                        #     'width_ratios': [1, 1,1,1.25]
                        #     # 'height_ratios': [1, 1,1,1]
                    #     })

for p in range(len(par_ind)):
    for es_i in range(len(ana_list)):
        Data_folder_IC = Data_folder+"SD"+str(ana_list[es_i])+"\\"
        results_files = [f for f in os.listdir(Data_folder_IC) if os.path.isfile(os.path.join(Data_folder_IC, f))]
        for n_t in range(len(range_ntrials)):
            for n_p in range(len(range_npp)):
                ntrials = range_ntrials[n_t]
                npp = range_npp[n_p]
                for file in results_files:
                    if criterion == "GD":
                        prefix_to_match = 'PowerGD{}P{}SD{}T{}N{}M{}ES'.format(par_ind[p],np.round(s_pooled,2),ntrials, 
                                                                                        npp, nreps,np.round(ana_list[es_i],2))
                    elif criterion == "EC": 
                        prefix_to_match = 'PowerEC{}P{}SD{}TC{}T{}N{}M'.format(par_ind[p],s_pooled, ana_list[es_i], ntrials,
                                                                                        npp, nreps)    
                    elif criterion == "IC": 
                        prefix_to_match = 'PowerIC{}T{}N{}M'.format(ntrials,npp, nreps)

                    if file.startswith(prefix_to_match) and file.endswith('.csv'):  # check prefix
                        
                        PowerFile_path = Data_folder_IC+file


                        PowerResults = pd.read_csv(PowerFile_path, delimiter = ',')[p_list[p]].dropna(axis = 0)

                        heatmap_p[n_t,n_p] = PowerResults.iloc[0]

            
        Power_AllData = (heatmap_p,)



        fontsize = 20
        xylabels = ['participants','trials']
        xticklabels = range_npp
        yticklabels = range_ntrials
        y_visible = 1

        if criterion == "GD":
            fig.suptitle("Pr(T-statistic > tau) with p-value threshold = 0.05 ".format(par_ind[p]), fontsize=fontsize)
        elif criterion == "EC":  
            fig.suptitle("Pr(External correlation > tau) with p-value threshold = 0.05 ".format(par_ind[p]), fontsize=fontsize)
        elif criterion == "IC":  
            fig.suptitle("Pr(Internal correlation > 0.8) ".format(par_ind[p]), fontsize=fontsize)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sns.set(font_scale=1.4)

        for i_p in range(len(par_ind)):
            data=pd.DataFrame(Power_AllData[i_p])

            if criterion == "GD":
                title = " Cohen's d = {}".format(ana_list[es_i])
            elif criterion == "EC":  
                title = " True correlation = {}".format(ana_list[es_i])
            elif criterion == "IC":  
                title = " SD of {} = {}".format(pn,ana_list[es_i])


            ax,figure = plot_MultiHeatmap(axs[es_i],es_i,len(ana_list),data,norm,title,xticklabels,yticklabels,xylabels)

plt.show()
print()
            # fig[0]=sns.heatmap(data,
            #                     xticklabels=xticklabels,
            #                     yticklabels=yticklabels,
            #                     annot=True,
            #                     cbar=colbar, 
            #                     ax = ax_fi,
            #                     cmap = 'Blues',
            #                     norm=norm
            #                 )
                
            # ax.set_title("power of parameter {}".format(p_list[i_p]), fontsize=18)
            # ax.set_xlabel(xylabels[0])  # x轴标题
            # ax.set_ylabel(xylabels[1])
            # ax.axes.yaxis.set_visible(y_visible)
            # ax.invert_yaxis()
            # figure = axs[0].get_figure()
        #     plt.show()
            
        #     axs[1]=sns.heatmap(c_data,
        #                         xticklabels=xticklabels,
        #                         yticklabels=yticklabels,
        #                         annot=True,
        #                         cbar=colbar, 
        #                         cmap = 'Blues',
        #                         norm=norm
        #                     )
                
        #     axs[1].set_title("conventional power of parameter {}".format(p_list[i_p]), fontsize=18)
        #     axs[1].set_xlabel(xylabels[0])  # x轴标题
        #     axs[1].set_ylabel(xylabels[1])
        #     axs[1].axes.yaxis.set_visible(y_visible)
        #     axs[1].invert_yaxis()
        #     figure = axs[1].get_figure()

        #     plt.show()

        # plt.show()
        # a =0