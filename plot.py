import os,sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ssms
# sys.path.append(r"D:\horiz\IMPORTANT\0study_graduate\Pro_COMPASS\COMPASS_DDM\results\test3")

plot_heatmap = 0
plot_single_setting = 1

ResultPath = "results\\test3"
DDM_id = "ddm"
tau = 0.8
nreps = 20
range_ntrials = [60]
range_npp = [20]
p_list = ['v','a','z','t']

heatmap_v = np.zeros((len(range_ntrials),len(range_npp)))
heatmap_a =  np.zeros((len(range_ntrials),len(range_npp)))
heatmap_z =  np.zeros((len(range_ntrials),len(range_npp)))
heatmap_t =  np.zeros((len(range_ntrials),len(range_npp)))

for n_t in range(len(range_ntrials)):
    for n_p in range(len(range_npp)):
        ntrials = range_ntrials[n_t]
        npp = range_npp[n_p]

        Parameters = ssms.config.model_config[DDM_id]["params"]
        OutputFile_name = 'OutputIC{}T{}N{}M.csv'.format(ntrials,npp, nreps)
        OutputFile_path = os.path.join(os.getcwd(), ResultPath, OutputFile_name)
        OutputResults = pd.read_csv(OutputFile_path, delimiter = ',')

        PowerFile_name = 'PowerIC{}T{}N{}M.csv'.format(ntrials,npp, nreps)
        PowerFile_path = os.path.join(os.getcwd(), ResultPath, PowerFile_name)
        PowerResults = pd.read_csv(PowerFile_path, delimiter = ',')[Parameters].dropna(axis = 0)


        # tau = OutputResults['tau'].values[-1]
        if plot_single_setting:
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

        if plot_heatmap:
            heatmap_v[n_t,n_p] = PowerResults['v'][0]
            heatmap_a[n_t,n_p] = PowerResults['a'][0]
            heatmap_z[n_t,n_p] = PowerResults['z'][0]
            heatmap_t[n_t,n_p] = PowerResults['t'][0]
            

Power_AllData = (heatmap_v,heatmap_a,heatmap_z,heatmap_t)


# plot heat map    
if plot_heatmap:
    fontsize = 20
    fig, axs = plt.subplots(1,len(p_list), 
                            gridspec_kw={
                                'width_ratios': [1, 1,1,1.25]
                                # 'height_ratios': [1, 1,1,1]
                                })

    fig.suptitle("Pr(internal correlation >= {}) with Nreps = {}".format(tau,nreps), fontsize=fontsize)
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
            power_plot = plot_MultiHeatmap(axs[i_p],i_p,[0,len(p_list)-1],data,title = "power of parameter {}".format(p_list[i_p]),
                                        norm = norm,
                            xticklabels = range_npp,yticklabels = range_ntrials,
                            xylabels = ['participants','trials'], t = 0.5)



    plt.show()

