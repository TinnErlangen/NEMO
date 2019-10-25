import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import stats

plt.ion() #this keeps plots interactive

base_dir = "D:/NEMO_analyses/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
subjs = ["nc_NEM_10"]

for subj in subjs:
    #load resting state measurement of a subjects
    rest = mne.read_epochs(proc_dir+subj+"_1_ica-epo.fif")
    # produce a layout that we use for sensor plotting; same for all conditions
    layout = mne.find_layout(rest.info)
    mag_names = [rest.ch_names[p] for p in mne.pick_types(rest.info, meg=True)]
    layout.names = mag_names
    #load tone baseline (run 2) of a subject
    tonbas = mne.read_epochs(proc_dir+subj+"_2_ica-epo.fif")
    #load run 3 and 4 of a subject
    epo_a = mne.read_epochs(proc_dir+subj+"_3_ica-epo.fif")
    epo_b = mne.read_epochs(proc_dir+subj+"_4_ica-epo.fif")
    #interpolate bad n_channels
    rest.interpolate_bads()
    tonbas.interpolate_bads()
    epo_a.interpolate_bads()
    epo_b.interpolate_bads()
    #override coil positions of block b with those of block a for concatenation (sensor level only!!)
    epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
    #concatenate data of both blocks
    epo = mne.concatenate_epochs([epo_a,epo_b])
    #separate positive and negative condition epochs
    pos = epo['positive']
    neg = epo['negative']

    #equalize number of epochs per condition by randomly dropping excess epochs
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


    #do psd freq analysis on each rest, tonbas, pos/neg, get psds and freqs
    rest_psds, rest_freqs = mne.time_frequency.psd_multitaper(rest,fmin=1,fmax=40,bandwidth=1)
    ton_psds, ton_freqs = mne.time_frequency.psd_multitaper(tonbas,fmin=1,fmax=40,bandwidth=1)
    pos_psds, pos_freqs = mne.time_frequency.psd_multitaper(pos,fmin = 1,fmax = 40,bandwidth =1)
    neg_psds, neg_freqs = mne.time_frequency.psd_multitaper(neg,fmin = 1,fmax = 40,bandwidth =1)

    #create PSD-to-Evoked objects for each condition
    rest_psd = np.mean(rest_psds,axis=0)
    rest.pick_types(meg=True)
    restev = rest.average()
    rest_fev = mne.EvokedArray(rest_psd,restev.info,comment="rest")
    rest_fev.times = rest_freqs
    ton_psd = np.mean(ton_psds,axis=0)
    tonbas.pick_types(meg=True)
    tonev = tonbas.average()
    ton_fev = mne.EvokedArray(ton_psd,tonev.info,comment="tonbas")
    ton_fev.times = ton_freqs
    pos_psd = np.mean(pos_psds,axis=0)
    pos.pick_types(meg=True)
    posev = pos.average()
    pos_fev = mne.EvokedArray(pos_psd,posev.info,comment="pos")
    pos_fev.times = pos_freqs
    neg_psd = np.mean(neg_psds,axis=0)
    neg.pick_types(meg=True)
    negev = neg.average()
    neg_fev = mne.EvokedArray(neg_psd,negev.info,comment="neg")
    neg_fev.times = neg_freqs
    # plot & compare the PSDs per condition over channel layout
    # colors = "black","grey","green","red"
    # mne.viz.plot_evoked_topo([rest_fev,ton_fev,pos_fev,neg_fev],layout=layout,color=colors)

    # create evoked condition comparisons
    ton_minus_rest = mne.combine_evoked([ton_fev,rest_fev],[1,-1])
    pos_minus_ton = mne.combine_evoked([pos_fev,ton_fev],[1,-1])
    neg_minus_ton = mne.combine_evoked([neg_fev,ton_fev],[1,-1])
    neg_minus_pos = mne.combine_evoked([neg_fev,pos_fev],[1,-1])

    # plot differences from pos/neg to tone baseline, and tone vs. resting state
    ton_minus_rest.plot(window_title="ToneBaseline - RestingState",spatial_colors=True)
    ton_minus_rest.plot_topomap(times=11,average=6.0,title="ToneBaseline - RestingState",cmap='RdBu_r')
    pos_minus_ton.plot(window_title="Positive - ToneBaseline",spatial_colors=True)
    pos_minus_ton.plot_topomap(times=11,average=6.0,title="Positive - ToneBaseline",cmap='RdBu_r')
    neg_minus_ton.plot(window_title="Negative - ToneBaseline",spatial_colors=True)
    neg_minus_ton.plot_topomap(times=11,average=6.0,title="Negative - ToneBaseline",cmap='RdBu_r')
    neg_minus_pos.plot(window_title="Negative - Positive",spatial_colors=True)
    neg_minus_pos.plot_topomap(times=11,average=6.0,title="Negative - Positive",cmap='RdBu_r')

    #get arrays of alpha trials x channels for N/P conditions for t-permutation-test and plotting topomap
    neg_A = np.mean(neg_psds[:,:,13:26],axis=2)
    pos_A = np.mean(pos_psds[:,:,13:26],axis=2)
    neg_minus_pos_A = neg_A - pos_A
    T_obs,p_vals,H0 = mne.stats.permutation_t_test(neg_minus_pos_A)
    NmP_t_evoked = mne.EvokedArray(T_obs[:,np.newaxis],restev.info,tmin=0,comment="t-values Negative minus Positive")
    NmP_t_evoked.plot_topomap(times=[0],scalings=1,layout=layout,cmap='RdBu_r',units='t Value')
    NmP_p_evoked = mne.EvokedArray(-np.log10(p_vals)[:,np.newaxis],restev.info,tmin=0,comment="p-values Negative minus Positive")
    mask = p_vals[:,np.newaxis] <= 0.05
    NmP_p_evoked.plot_topomap(times=[0],scalings=1,layout=layout,cmap='Reds',units='-log10(p)',mask=mask)

    # rest_alpha = np.mean(np.mean(rest_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # ton_alpha = np.mean(np.mean(ton_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # pos_alpha = np.mean(np.mean(pos_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # neg_alpha = np.mean(np.mean(neg_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
