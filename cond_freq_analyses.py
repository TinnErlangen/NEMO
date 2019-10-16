
import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
subjs = ["nc_NEM_37"]
#subjs = ["nc_NEM_25"]
runs = ["3","4"]
#runs=["1","2"]
runs = ["4"]

for subj in subjs:
    for run in runs:
        #load run 3 of a subject
        epo = mne.read_epochs(proc_dir+subj+"_"+run+"_ica-epo.fif")
        #separate positive and negative condition epochs
        pos = epo['positive']
        neg = epo['negative']
        #plot a topomap for alpha for each pos/neg  (later for other bands)
        # pos.plot_psd_topomap(bands=[(8,13,'Alpha')],vmin=-255,vmax=-235,bandwidth=0.8,cmap='RdYlBu_r')
        # neg.plot_psd_topomap(bands=[(8,13,'Alpha')],vmin=-255,vmax=-235,bandwidth=0.8,cmap='RdYlBu_r')
        #do psd freq analysis on each pos/neg, get psds and freqs
        pos_psds, pos_freqs = mne.time_frequency.psd_multitaper(pos,fmin = 1,fmax = 40,bandwidth =0.8)
        neg_psds, neg_freqs = mne.time_frequency.psd_multitaper(neg,fmin = 1,fmax = 40,bandwidth =0.8)
        #do a psd to evo transform for both
        pos_psd = np.mean(pos_psds,axis=0)
        pos.pick_types(meg=True)
        posev = pos.average()
        posevpsd = mne.EvokedArray(pos_psd,posev.info)
        posevpsd.times = pos_freqs
        neg_psd = np.mean(neg_psds,axis=0)
        neg.pick_types(meg=True)
        negev = neg.average()
        negevpsd = mne.EvokedArray(neg_psd,negev.info)
        negevpsd.times = neg_freqs
        # create a difference evo neg-pos, and plot, and plot plot_psd_topomap
        neg_minus_pos = mne.combine_evoked([negevpsd,posevpsd],[1,-1])

        ## PLOTTING ##

        # plot 1st to find appropriate scale for subject
        # posevpsd.plot(window_title="Positive",spatial_colors=True)
        # posevpsd.plot_topomap(times=10.5,average=5.0,title="Positive",cmap='Reds')
        # posevpsd.plot_joint(times='peaks')
        # negevpsd.plot(window_title="Negative",spatial_colors=True)
        # negevpsd.plot_topomap(times=10.5,average=5.0,title="Negative",cmap='Reds')
        # negevpsd.plot_joint(times='peaks')
        # neg_minus_pos.plot(window_title="Negative - Positive",spatial_colors=True)
        # neg_minus_pos.plot_topomap(times=10.5,average=5.0,title="Negative - Positive",cmap='RdBu_r')
        # neg_minus_pos.plot_joint(times='peaks')

        # plot again with set/constant scale
        # posevpsd.plot(window_title="Positive",spatial_colors=True,ylim=dict(mag=[0,2.5e-09]))
        # posevpsd.plot_topomap(times=10.5,average=5.0,title="Positive",vmin=0,vmax=0.7e-09,cmap='Reds')
        # negevpsd.plot(window_title="Negative",spatial_colors=True,ylim=dict(mag=[0,2.5e-09]))
        # negevpsd.plot_topomap(times=10.5,average=5.0,title="Negative",vmin=0,vmax=0.7e-09,cmap='Reds')
        # neg_minus_pos.plot(window_title="Negative - Positive",spatial_colors=True,ylim=dict(mag=[-1.5e-09,1.5e-09]))
        # neg_minus_pos.plot_topomap(times=10.5,average=5.0,title="Negative - Positive",vmin=-3e-10,vmax=3e-10,cmap='RdBu_r')
