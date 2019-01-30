#step 4 - do ICA on annotated epoch files

import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]
runs = ["2","3","4"]

for sub in subjs:
    for run in runs:
        #load the annotated epoch file
        mepo = mne.read_epochs(proc_dir+sub+"_"+run+"_m-epo.fif")
        #define the ica and fit it onto epoch file
        ica = mne.preprocessing.ICA(n_components=0.95,max_iter=500,method="picard") #parameters for ica
        ica.fit(mepo)
        #save the ica result in its own file
        ica.save(proc_dir+sub+"_"+run+"_mepo-ica.fif")
