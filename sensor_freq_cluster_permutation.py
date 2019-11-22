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

NEM_data = np.zeros((2,78,248,8))

for subj in subjs:

    #load tone baseline (run 2) of a subject
    #tonbas = mne.read_epochs(proc_dir+subj+"_2_ica-epo.fif")
    #load run 3 and 4 of a subject
    epo_a = mne.read_epochs(proc_dir+subj+"_3_ica-epo.fif")
    epo_b = mne.read_epochs(proc_dir+subj+"_4_ica-epo.fif")
    # produce a layout that we use for sensor plotting; same for all conditions
    layout = mne.find_layout(epo_a.info)
    mag_names = [epo_a.ch_names[p] for p in mne.pick_types(epo_a.info, meg=True)]
    layout.names = mag_names
    #interpolate bad n_channels
    #tonbas.interpolate_bads()
    epo_a.interpolate_bads()
    epo_b.interpolate_bads()
    #override coil positions of block b with those of block a for concatenation (sensor level only!!)
    epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
    #concatenate data of both blocks
    epo = mne.concatenate_epochs([epo_a,epo_b])
    # equalize event counts of experiment -- if wanted !! -- choose one or none of the following two options
    # epo.equalize_event_counts(event_ids=['positive','negative'])
    # epo.equalize_event_counts(event_ids=['positive/r1','positive/r2','positive/s1','positive/s2','negative/r1','negative/r2','negative/s1','negative/s2'])
    # make frequency analyses for all 8 conditions (2 emos * 4 tones)
    neg1_psds, neg1_freqs = mne.time_frequency.psd_multitaper(epo['negative/r1'],fmin = 1,fmax = 40,bandwidth =1)
    neg2_psds, neg2_freqs = mne.time_frequency.psd_multitaper(epo['negative/r2'],fmin = 1,fmax = 40,bandwidth =1)
    neg3_psds, neg3_freqs = mne.time_frequency.psd_multitaper(epo['negative/s1'],fmin = 1,fmax = 40,bandwidth =1)
    neg4_psds, neg4_freqs = mne.time_frequency.psd_multitaper(epo['negative/s2'],fmin = 1,fmax = 40,bandwidth =1)
    pos1_psds, pos1_freqs = mne.time_frequency.psd_multitaper(epo['positive/r1'],fmin = 1,fmax = 40,bandwidth =1)
    pos2_psds, pos2_freqs = mne.time_frequency.psd_multitaper(epo['positive/r2'],fmin = 1,fmax = 40,bandwidth =1)
    pos3_psds, pos3_freqs = mne.time_frequency.psd_multitaper(epo['positive/s1'],fmin = 1,fmax = 40,bandwidth =1)
    pos4_psds, pos4_freqs = mne.time_frequency.psd_multitaper(epo['positive/s2'],fmin = 1,fmax = 40,bandwidth =1)
    # rearrange the PSD arrays to get trials*freqs*channels
    neg1_freq = np.swapaxes(neg1_psds,1,2)
    neg2_freq = np.swapaxes(neg2_psds,1,2)
    neg3_freq = np.swapaxes(neg3_psds,1,2)
    neg4_freq = np.swapaxes(neg4_psds,1,2)
    pos1_freq = np.swapaxes(pos1_psds,1,2)
    pos2_freq = np.swapaxes(pos2_psds,1,2)
    pos3_freq = np.swapaxes(pos3_psds,1,2)
    pos4_freq = np.swapaxes(pos4_psds,1,2)
    # calculate the means across trials for each (resulting in a flat array each)
    neg1_F = np.mean(neg1_freq,axis=0)
    neg2_F = np.mean(neg2_freq,axis=0)
    neg3_F = np.mean(neg3_freq,axis=0)
    neg4_F = np.mean(neg4_freq,axis=0)
    pos1_F = np.mean(pos1_freq,axis=0)
    pos2_F = np.mean(pos2_freq,axis=0)
    pos3_F = np.mean(pos3_freq,axis=0)
    pos4_F = np.mean(pos4_freq,axis=0)
    # stack them along a new last dimension *conditions
    Fs_by_Conds = np.stack((neg1_F,neg2_F,neg3_F,neg4_F,pos1_F,pos2_F,pos3_F,pos4_F),axis=-1)

    NEM_data = np.append(NEM_data,[Fs_by_Conds],axis=0)

NEM_data = np.delete(NEM_data,[0,1],axis=0)
print(NEM_data.shape)

chan_connectivity, chans = mne.channels.find_ch_connectivity(epo.info,ch_type='mag')

NEM_anov = [np.squeeze (x) for x in np.split(NEM_data,8,axis=-1)]

t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test([NEM_data],stat_fun = None,connectivity=chan_connectivity, )
