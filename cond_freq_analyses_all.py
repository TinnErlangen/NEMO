
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import stats

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_37"]
#subjs = ["nc_NEM_25"]

logfile = base_dir+"NEMO_alpha_pos-neg.txt"
with open(logfile,"w") as file:

    # create SUPER DATAFRAME
    subjects = {'Subjects':subjs}
    NEMO = pd.DataFrame(subjects,columns=['Subjects'])
    P_Alpha = []
    N_Alpha = []

    for subj in subjs:
        file.write("\n\n"+subj+"\n\n")
        #load run 3 and 4 of a subject
        epo_a = mne.read_epochs(proc_dir+subj+"_3_ica-epo.fif")
        epo_b = mne.read_epochs(proc_dir+subj+"_4_ica-epo.fif")
        # check bad channels and make them equal for concatenation
        bads = []
        for bad_chan in epo_a.info['bads']:
            bads.append(bad_chan)
        for bad_chan in epo_b.info['bads']:
            bads.append(bad_chan)
        epo_a.info['bads'] = bads
        epo_b.info['bads'] = bads
        #override coil positions of block b with those of block a for concatenation (sensor level only!!)
        epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
        #concatenate data of both blocks
        epo = mne.concatenate_epochs([epo_a,epo_b])
        #separate positive and negative condition epochs
        pos = epo['positive']
        neg = epo['negative']
        #equalize number of epochs per condition by random dropping excess epochs
        len_diff = len(pos.events)-len(neg.events)
        if len_diff > 0:
            drops = random.sample(range(len(pos.events)),k=len_diff)
            drops.sort(reverse=True)
            for drop in drops:
                pos.drop([drop])
        if len_diff < 0:
            drops = random.sample(range(len(neg.events)),k=abs(len_diff))
            drops.sort(reverse=True)
            for drop in drops:
                neg.drop([drop])

        #plot a topomap for alpha for each pos/neg  (later for other bands)
        # pos.plot_psd_topomap(bands=[(8,13,'Alpha')],vmin=-255,vmax=-235,bandwidth=0.8,cmap='RdYlBu_r')
        # neg.plot_psd_topomap(bands=[(8,13,'Alpha')],vmin=-255,vmax=-235,bandwidth=0.8,cmap='RdYlBu_r')
        #do psd freq analysis on each pos/neg, get psds and freqs
        pos_psds, pos_freqs = mne.time_frequency.psd_multitaper(pos,fmin = 1,fmax = 40,bandwidth =1)
        neg_psds, neg_freqs = mne.time_frequency.psd_multitaper(neg,fmin = 1,fmax = 40,bandwidth =1)
        #get arrays of alpha psds per condition
        pos_alpha = np.mean(np.mean(pos_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
        neg_alpha = np.mean(np.mean(neg_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
        # compute descriptives and single subject t-test and log into file
        file.write("Alpha Positive\n")
        file.write(str(stats.describe(pos_alpha)))
        file.write("\nAlpha Negative\n")
        file.write(str(stats.describe(neg_alpha)))
        file.write("\nAlpha T-test Pos-Neg\n")
        file.write(str(stats.ttest_rel(pos_alpha,neg_alpha)))
        # add condition alpha means to values for super dataframe
        P_Alpha.append(np.mean(pos_alpha))
        N_Alpha.append(np.mean(neg_alpha))

    # add single subject means as variables to super dataframe
    NEMO['P_Alpha'] = P_Alpha
    NEMO['N_Alpha'] = N_Alpha
    # make group stats
    file.write("\n\nGROUP STATISTICS\n\n")
    file.write("Alpha Positive\n")
    file.write(str(stats.describe(NEMO['P_Alpha'])))
    file.write("\nAlpha Negative\n")
    file.write(str(stats.describe(NEMO['N_Alpha'])))
    file.write("\nAlpha T-test Pos-Neg\n")
    file.write(str(stats.ttest_rel(NEMO['P_Alpha'],NEMO['N_Alpha'])))




            # #do a psd to evo transform for both
            # pos_psd = np.mean(pos_psds,axis=0)
            # pos.pick_types(meg=True)
            # posev = pos.average()
            # posevpsd = mne.EvokedArray(pos_psd,posev.info)
            # posevpsd.times = pos_freqs
            # neg_psd = np.mean(neg_psds,axis=0)
            # neg.pick_types(meg=True)
            # negev = neg.average()
            # negevpsd = mne.EvokedArray(neg_psd,negev.info)
            # negevpsd.times = neg_freqs
            # # create a difference evo neg-pos, and plot, and plot plot_psd_topomap
            # neg_minus_pos = mne.combine_evoked([negevpsd,posevpsd],[1,-1])

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
