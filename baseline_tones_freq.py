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

for subj in subjs:
    tonbas = mne.read_epochs(proc_dir+subj+"_2_ica-epo.fif")
    ton_psds, ton_freqs = mne.time_frequency.psd_multitaper(tonbas,fmin=1,fmax=40,bandwidth=1)
    ton_alpha = np.mean(np.mean(ton_psds[...,13:26],axis=2),axis=-1)
    stats.describe(ton_alpha)
