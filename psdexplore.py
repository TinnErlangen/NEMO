# exploring the Alpha power in resting state

import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16"]
subjs = ["nc_NEM_10"]

for sub in subjs:

# load cleaned data for resting state, tone baseline, experiment blocks A+B
    rest = mne.read_epochs(proc_dir+sub+"_1_ica-epo.fif")
    ton = mne.read_epochs(proc_dir+sub+"_2_ica-epo.fif")
    exp3 = mne.read_epochs(proc_dir+sub+"_3_ica-epo.fif")
    exp4 = mne.read_epochs(proc_dir+sub+"_4_ica-epo.fif")

    # explore resting state Alpha
    rest.plot_psd()  # note down alpha peak freqs and channels from here
    rest.plot_psd_topomap()

    # explore tonebase Alpha
    ton.plot_psd()  # note down alpha peak freqs and channels from here
    ton.plot_psd_topomap()

    # explore experimental alpha positive vs. negative - per Block (then combined?)
    exp3.plot_psd()  # note down alpha peak freqs and channels from here
    exp3.plot_psd_topomap()
    exp4.plot_psd()  # note down alpha peak freqs and channels from here
    exp4.plot_psd_topomap()
