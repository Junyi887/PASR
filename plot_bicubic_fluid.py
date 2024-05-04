import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator

correlation_tri_DT = np.load("/pscratch/sd/j/junyi012/PASR/vorticity_correlation_tri_DT.npy")
correlation_FNO_DT = np.load("/pscratch/sd/j/junyi012/PASR/vorticity_correlation_FNO_DT.npy")
correlation_ConvLSTM_DT = np.load("vorticity_correlation_convL_DT.npy")
correlation_our_DT = np.load("vorticity_correlation_DT.npy")
correlation_tri_RBC = np.load("vorticity_correlation_tri_RBC.npy")
correlation_FNO_RBC = np.load("vorticity_correlation_FNO_RBC.npy")
correlation_ConvLSTM_RBC = np.load("vorticity_correlation_convL_RBC.npy")
correlation_our_RBC = np.load("vorticity_correlation_RBC.npy")

x_spectrum_DT =  np.load("/pscratch/sd/j/junyi012/PASR/energy_spectrum_realsize_pred_DT.npy")
y_spectrum_DT =  np.load("energy_spectrum_EK_avsphr_pred_DT.npy")
x_spectrum_tri_DT = np.load("/pscratch/sd/j/junyi012/PASR/energy_spectrum_realsize_pred_tri_DT.npy")
y_spectrum_tri_DT = np.load("energy_spectrum_EK_avsphr_pred_tri_DT.npy")
x_spectrum_FNO_DT = np.load("/pscratch/sd/j/junyi012/PASR/energy_spectrum_realsize_pred_FNO_DT.npy")
y_spectrum_FNO_DT = np.load("energy_spectrum_EK_avsphr_pred_FNO_DT.npy")
x_spectrum_ConvLSTM_DT = np.load("energy_spectrum_realsize_pred_convL_DT.npy")
y_spectrum_ConvLSTM_DT = np.load("energy_spectrum_EK_avsphr_pred_convL_DT.npy")
y_specturm_truth_DT = np.load("energy_spectrum_EK_avsphr_truth_DT.npy")

spectrum_x_RBC =  np.load("energy_spectrum_realsize_pred_RBC.npy")
spectrum_y_RBC =  np.load("energy_spectrum_EK_avsphr_pred_RBC.npy")
spectrum_x_tri_RBC = np.load("energy_spectrum_realsize_pred_tri_RBC.npy")
spectrum_y_tri_RBC = np.load("energy_spectrum_EK_avsphr_pred_tri_RBC.npy")
spectrum_x_FNO_RBC = np.load("energy_spectrum_realsize_pred_FNO_RBC.npy")
spectrum_y_FNO_RBC = np.load("energy_spectrum_EK_avsphr_pred_FNO_RBC.npy")
spectrum_x_ConvLSTM_RBC = np.load("energy_spectrum_realsize_pred_convL_RBC.npy")
spectrum_y_ConvLSTM_RBC = np.load("energy_spectrum_EK_avsphr_pred_convL_RBC.npy")
spectrum_y_truth_RBC = np.load("energy_spectrum_EK_avsphr_truth_RBC.npy")
def get_config(data_name):
    if data_name == "DT":
        x_bound = 25
        zoom_in_localtion = [0.2, 0.2, 0.4, 0.2]
        x1, x2, y1, y2 = 4.9, 6.8, 1e-3, 2e-3  # subregion of the original image
        save_batch = 3
        xticks = [8,10]
        yticks = [1e-5,1e-4]
        yticks_label = [r'$10^{-5}$' ,r'10^{-4}$']
        y_bound = [1e-6,1e-1]
        xticks_label = [r'$6 \times 10^0$' ,r'$10^1$']
        batch_start = 15
        batch_end = 21
        zoom_in_factor = 6
    if data_name =="RBC":
        x_bound = 25
        zoom_in_localtion = [0.17, 0.29, 0.49, 0.2]
        x1, x2, y1, y2 = 7.7, 9.3, 1e-4, 3e-4   # subregion of the original image 
        save_batch = 9
        y_bound = [1e-6,1e-1]
        xticks = [x1,x2]
        xticks_label = [r'$9 \times 10^0$' ,r'$10^1$']
        yticks = [y1,y2]
        yticks_label = [r'$10^{-5}$' ,r'$10^{-4}$']
        batch_start = 0
        batch_end = 20
        zoom_in_factor = 6 
    return x_bound, zoom_in_localtion, x1, x2, y1, y2, save_batch, xticks, yticks, y_bound, xticks_label, yticks_label, batch_start, batch_end, zoom_in_factor
fig, ax = plt.subplots(2, 2, figsize=(5.4*1.2, 4.8*1.2))
# color_profile = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
color_profile = sns.color_palette('YlGnBu_r', n_colors=4)
alpha = 1
markersize =30
fontsize= 8
ax[0,0].set_xticks(np.arange(0,correlation_FNO_DT.shape[0],5),[0,None,0.5,None,1])
ax[0,0].set_yticks(np.arange(0.85,1,0.1))
ax[0,1].set_xticks(np.arange(0,correlation_FNO_RBC.shape[0],5),[0,None,0.5,None,1])
ax[0,1].set_yticks(np.arange(0.85,1,0.1))

ax[0,0].set_xticklabels(["0",None,"0.5",None,"1"],fontsize=fontsize)
ax[0,1].set_xticklabels(["0",None,"0.5",None,"1"],fontsize=fontsize)
ax[0,0].set_yticklabels(np.arange(0.85,1,0.1),fontsize=fontsize)
ax[0,1].set_yticklabels(np.arange(0.85,1,0.1),fontsize=fontsize)

ax[0,1].set_yticks([])
ax[1,1].set_yticks([])

ax[0,0].set_ylabel("Vorticity correlation",fontsize=fontsize)
ax[0,0].set_xlabel("Time",fontsize=fontsize)
ax[0,1].set_xlabel("Time",fontsize=fontsize)
ax[0,0].set_ylim(0.85,1.01)
ax[0,1].set_ylim(0.85,1.01)
ax[0,0].set_title("DT",fontsize=fontsize)
ax[0,1].set_title("RBC",fontsize=fontsize)
ax[0,0].plot(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_our_DT,color=color_profile[0],label="Ours")
ax[0,0].plot(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_ConvLSTM_DT,color = color_profile[1],label="ConvLSTM",alpha=alpha)
ax[0,0].plot(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_FNO_DT,color = color_profile[2],label="FNO",alpha=alpha)
ax[0,0].plot(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_tri_DT,color = color_profile[3],label="TriLinear",alpha=alpha)
ax[0,0].scatter(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_our_DT,color=color_profile[0],marker =".",s=markersize)
ax[0,0].scatter(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_ConvLSTM_DT,color = color_profile[1],marker =".",s=markersize)
ax[0,0].scatter(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_FNO_DT,color = color_profile[2],marker= ".",s=markersize)
ax[0,0].scatter(np.arange(0,correlation_ConvLSTM_DT.shape[0],1),correlation_tri_DT,color = color_profile[3],marker = ".",s=markersize)
ax[0,0].axhline(y = 0.95, color = 'k', linestyle = 'dashed',alpha=0.5,label="95% reference line") 
ax[0,1].plot(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_our_RBC,color=color_profile[0],label="Ours")
ax[0,1].plot(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_ConvLSTM_RBC,color = color_profile[1],label="ConvLSTM",alpha=alpha)
ax[0,1].plot(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_FNO_RBC,color = color_profile[2],label="FNO",alpha=alpha)
ax[0,1].plot(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_tri_RBC,color = color_profile[3],label="TriLinear",alpha=alpha)
ax[0,1].scatter(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_our_RBC,color=color_profile[0],marker =".",s=markersize)
ax[0,1].scatter(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_ConvLSTM_RBC,color = color_profile[1],marker =".",s=markersize)
ax[0,1].scatter(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_FNO_RBC,color = color_profile[2],marker= ".",s=markersize)
ax[0,1].scatter(np.arange(0,correlation_FNO_RBC.shape[0],1),correlation_tri_RBC,color = color_profile[3],marker = ".",s=markersize)
# set y axis off
ax[0,1].axhline(y = 0.95, color = 'k', linestyle = 'dashed',alpha=0.5,label="95% reference line") 


ax[0,1].legend(loc='lower left',fontsize=fontsize,frameon=False)

ax[1,0].loglog(range(len(y_spectrum_DT)),y_spectrum_DT,color=color_profile[0],label="Ours")
ax[1,0].loglog(range(len(y_spectrum_ConvLSTM_DT)),y_spectrum_ConvLSTM_DT,color = color_profile[1],label="ConvLSTM",alpha=alpha)
ax[1,0].loglog(range(len(y_spectrum_FNO_DT)),y_spectrum_FNO_DT,color = color_profile[2],label="FNO",alpha=alpha)
ax[1,0].loglog(range(len(y_spectrum_tri_DT)),y_spectrum_tri_DT,color = color_profile[3],label="TriLinear",alpha=alpha)
xbound01, zoom_in_localtion01, x1, x2, y1, y2, save_batch, xticks, yticks, y_bound, xticks_label, yticks_label, batch_start, batch_end, zoom_in_factor = get_config("DT")
ax[1,0].set_xlim(left=1, right=xbound01)
ax[1,0].set_ylim(bottom=y_bound[0], top=y_bound[1])

ax[1,0].set_ylabel(r"Energy E[k]",fontsize=fontsize)
ax[1,0].set_xlabel(r"Wavenumber k",fontsize=fontsize)
ax[1,1].set_xlabel(r"Wavenumber k",fontsize=fontsize)
ax[1,0].xaxis.set_tick_params(labelsize=fontsize)
ax[1,0].yaxis.set_tick_params(labelsize=fontsize)
ax[1,1].xaxis.set_tick_params(labelsize=fontsize)
axins0 = zoomed_inset_axes(ax[1,0], zoom_in_factor,loc='lower left')
_patch, pp1, pp2 = mark_inset(ax[1,0], axins0, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.5, color='k') 
axins0.loglog(range(len(y_specturm_truth_DT)),y_specturm_truth_DT,color='k',label="Truth")
axins0.loglog(range(len(y_spectrum_DT)),y_spectrum_DT,color=color_profile[0],label="Ours")
axins0.loglog(range(len(y_spectrum_ConvLSTM_DT)),y_spectrum_ConvLSTM_DT,color = color_profile[1],label="ConvLSTM",alpha=alpha)
axins0.loglog(range(len(y_spectrum_FNO_DT)),y_spectrum_FNO_DT,color = color_profile[2],label="FNO",alpha=alpha)
axins0.loglog(range(len(y_spectrum_tri_DT)),y_spectrum_tri_DT,color = color_profile[3],label="TriLinear",alpha=alpha)
axins0.set_xlim(x1, x2)
axins0.set_ylim(y1, y2)
axins0.xaxis.set_tick_params(labelbottom=False)
axins0.yaxis.set_tick_params(labelleft=False)
axins0.set_xticks([])
axins0.set_yticks([])
axins0.minorticks_off()

ax[1,1].loglog(range(len(spectrum_y_RBC)),spectrum_y_RBC,color=color_profile[0],label="Ours")
ax[1,1].loglog(range(len(spectrum_y_ConvLSTM_RBC)),spectrum_y_ConvLSTM_RBC,color = color_profile[1],label="ConvLSTM",alpha=alpha)
ax[1,1].loglog(range(len(spectrum_y_FNO_RBC)),spectrum_y_FNO_RBC,color = color_profile[2],label="FNO",alpha=alpha)
ax[1,1].loglog(range(len(spectrum_y_tri_RBC)),spectrum_y_tri_RBC,color = color_profile[3],label="TriLinear",alpha=alpha)
ax[1,1].yaxis.set_tick_params(labelleft=False)

xbound02, zoom_in_localtion02, x1, x2, y1, y2, save_batch, xticks, yticks, y_bound, xticks_label, yticks_label, batch_start, batch_end, zoom_in_factor = get_config("RBC")
ax[1,1].set_xlim(left=1, right=xbound02)
ax[1,1].set_ylim(bottom=y_bound[0], top=y_bound[1])

axins1 = zoomed_inset_axes(ax[1,1], zoom_in_factor,loc='lower left')
_patch, pp1, pp2 = mark_inset(ax[1,1], axins1, loc1=2, loc2=4, fc='none', ec='0.5', lw=1.0, color='k')
axins1.loglog(range(len(spectrum_y_truth_RBC)),spectrum_y_truth_RBC,color='k',label="Truth")
axins1.loglog(range(len(spectrum_y_RBC)),spectrum_y_RBC,color=color_profile[0],label="Ours")
axins1.loglog(range(len(spectrum_y_ConvLSTM_RBC)),spectrum_y_ConvLSTM_RBC,color = color_profile[1],label="ConvLSTM",alpha=alpha)
axins1.loglog(range(len(spectrum_y_FNO_RBC)),spectrum_y_FNO_RBC,color = color_profile[2],label="FNO",alpha=alpha)
axins1.loglog(range(len(spectrum_y_tri_RBC)),spectrum_y_tri_RBC,color = color_profile[3],label="TriLinear",alpha=alpha)
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.xaxis.set_tick_params(labelbottom=False)
axins1.yaxis.set_tick_params(labelleft=False)
axins1.set_xticks([])
axins1.set_yticks([])
axins1.minorticks_off()


ax[1,0].yaxis.set_major_locator(LogLocator(numticks=4))
ax[1,0].xaxis.set_major_locator(LogLocator(numticks=3))

# axins.set_yscale('log')
# axins.set_xscale('log')

# axins.set_xticks(xticks,['',''])
# axins.set_yticks(yticks,['',''])
# axins.xaxis.set_tick_params(labelbottom=False)
# axins.yaxis.set_tick_params(labelleft=False)
# axins.set_xticks([])
# axins.set_yticks([])
# axins.minorticks_off()

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig.savefig("bicubic_fluid.pdf",bbox_inches='tight')