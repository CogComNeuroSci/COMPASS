import os,sys
import pandas as pd
import seaborn as sns
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import ssms
def ImPlot_plot(title,x,xlabel,y,ylabel,type = "plot",titlesize=35,xylabelsize=25,labelsize = 25,linewidth = 5):
    # corr_coef = np.corrcoef(x, y)[0, 1]
    # coefficients = np.polyfit(x, y, 1)  # 线性回归，拟合一次多项式（一次直线）
    # poly = np.poly1d(coefficients)
    # x_fit = np.linspace(min(x), max(x))  # 创建拟合线的 x 值
    # y_fit = poly(x_fit) 

    plt.figure(figsize=(12, 12))
    plt.title(title,fontdict={"fontsize" : titlesize})
    if type == "plot":
        plt.plot(x,y,linewidth=linewidth)
    elif type == "scatter":
        plt.scatter(x,y,s=linewidth)
        # plt.plot(x_fit, y_fit, color='red', label='Regression line')  # 回归线


    plt.xlabel(xlabel,fontdict={"fontsize" : xylabelsize})
    plt.tick_params(labelsize=labelsize)
    plt.ylabel(ylabel,fontdict={"fontsize" : xylabelsize})
    return plt

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

# sys.path.append(r"D:\horiz\IMPORTANT\0study_graduate\Pro_COMPASS\COMPASS_DDM\results\test3")

plot_heatmap = 0
plot_single_setting = 1
plot_ProPow = 1

ResultPath = "results\\test_Cluster\\v_0.3"
DDM_id = "ddm"
range_ntrials = [100]
range_npp = [40]
tau = 0.8
nreps = 80

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
        out_cn = Parameters+["ACC","RT"]
        OutputResults = pd.read_csv(OutputFile_path, delimiter = ',')[out_cn].dropna(axis = 0)


        PowerFile_name = 'PowerIC{}T{}N{}M.csv'.format(ntrials,npp, nreps)
        PowerFile_path = os.path.join(os.getcwd(), ResultPath, PowerFile_name)
        PowerResults = pd.read_csv(PowerFile_path, delimiter = ',')[Parameters].dropna(axis = 0)


        # tau = OutputResults['tau'].values[-1]
        if plot_single_setting:
            for p in Parameters :
                    plt.figure(figsize=(12, 12))
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
        if plot_ProPow:
            for n_p in range(len(p_list)):
                x = OutputResults[p_list[n_p]].values
                y = OutputResults["ACC"].values
                plt = ImPlot_plot("Correlation between ACC \nand recovery of "+p_list[n_p],
                            x,p_list[n_p],y ,'ACC',
                            type = "scatter",titlesize=35,xylabelsize=25,labelsize = 25,linewidth = 20)
                sns.regplot(x=OutputResults[p_list[n_p]], y=OutputResults["ACC"])
                corr_coef, p_value = stats.pearsonr(x, y)
                annotation = f"Correlation coefficient: {corr_coef:.2f}\nP-value: {p_value:.4f}"
                plt.annotate(annotation, xy=(0.1, 0.85), xycoords='axes fraction', fontsize=20)

                file_name = '\ProPowIC{}T{}N{}M_{}.png'.format(ntrials,npp,nreps,p_list[n_p])
                plt.savefig(ResultPath+file_name,bbox_inches='tight')



        if plot_heatmap:
            heatmap_v[n_t,n_p] = PowerResults['v'][0]
            heatmap_a[n_t,n_p] = PowerResults['a'][0]
            heatmap_z[n_t,n_p] = PowerResults['z'][0]
            heatmap_t[n_t,n_p] = PowerResults['t'][0]
        



    plt.tick_params(labelsize=labelsize)

    plt.show()

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

