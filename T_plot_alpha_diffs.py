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
#subjs = ["nc_NEM_10"]

A_NmP = []

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

    #get N_alpha minus P_alpha array per person [n_channels] & append to all_subject_array
    neg_A = np.mean(neg_psds[:,:,13:26],axis=2)
    pos_A = np.mean(pos_psds[:,:,13:26],axis=2)
    neg_minus_pos_A = neg_A - pos_A
    neg_minus_pos_A_bychan = np.mean(neg_minus_pos_A,axis=0)
    A_NmP.append(neg_minus_pos_A_bychan)

A_NmP = np.array(A_NmP)
T_obs,p_vals,H0 = mne.stats.permutation_t_test(A_NmP)
rest.pick_types(meg=True)
restev = rest.average()
NmP_t_evoked = mne.EvokedArray(T_obs[:,np.newaxis],restev.info,tmin=0,comment="t-values Negative minus Positive")
NmP_t_evoked.plot_topomap(times=[0],scalings=1,layout=layout,cmap='RdBu_r',units='t Value')
NmP_p_evoked = mne.EvokedArray(-np.log10(p_vals)[:,np.newaxis],restev.info,tmin=0,comment="p-values Negative minus Positive")
mask = p_vals[:,np.newaxis] <= 0.05
NmP_p_evoked.plot_topomap(times=[0],scalings=1,layout=layout,cmap='Reds',vmin=0.,vmax=np.max,units='-log10(p)',mask=mask)
