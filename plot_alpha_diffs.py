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
    #load tone baseline (run 2) of a subject
    tonbas = mne.read_epochs(proc_dir+subj+"_2_ica-epo.fif")
    #load run 3 and 4 of a subject
    epo_a = mne.read_epochs(proc_dir+subj+"_3_ica-epo.fif")
    epo_b = mne.read_epochs(proc_dir+subj+"_4_ica-epo.fif")
    # check bad channels and make them equal for concatenation and comparisons
    bads = []
    for bad_chan in rest.info['bads']:
        bads.append(bad_chan)
    for bad_chan in tonbas.info['bads']:
        bads.append(bad_chan)
    for bad_chan in epo_a.info['bads']:
        bads.append(bad_chan)
    for bad_chan in epo_b.info['bads']:
        bads.append(bad_chan)
    rest.info['bads'] = bads
    tonbas.info['bads'] = bads
    epo_a.info['bads'] = bads
    epo_b.info['bads'] = bads
    #override coil positions of block b with those of block a for concatenation (sensor level only!!)
    epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
    #concatenate data of both blocks
    epo = mne.concatenate_epochs([epo_a,epo_b])
    #separate positive and negative condition epochs
    pos = epo['positive']
    neg = epo['negative']

    #do psd freq analysis on each rest, tonbas, pos/neg, get psds and freqs
    rest_psds, rest_freqs = mne.time_frequency.psd_multitaper(rest,fmin=1,fmax=40,bandwidth=1)
    ton_psds, ton_freqs = mne.time_frequency.psd_multitaper(tonbas,fmin=1,fmax=40,bandwidth=1)
    pos_psds, pos_freqs = mne.time_frequency.psd_multitaper(pos,fmin = 1,fmax = 40,bandwidth =1)
    neg_psds, neg_freqs = mne.time_frequency.psd_multitaper(neg,fmin = 1,fmax = 40,bandwidth =1)
    #get arrays of alpha psds per condition
    # rest_alpha = np.mean(np.mean(rest_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # ton_alpha = np.mean(np.mean(ton_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # pos_alpha = np.mean(np.mean(pos_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
    # neg_alpha = np.mean(np.mean(neg_psds[...,13:26],axis=2),axis=-1) # 13 freq values from 7.98 to 13.97
