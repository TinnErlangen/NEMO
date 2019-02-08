#step 4 - do ICA on annotated epoch files

import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]
#runs = ["1"]

for sub in subjs:
    for run in runs:
        #load the annotated epoch file
        mepo = mne.read_epochs(proc_dir+sub+"_"+run+"_m-epo.fif")
        mepo.info["bads"] += ["MRyA","MRyaA"] # add broken reference channels to bad channels list
        #define the ica for reference channels and fit it onto epoch file
        icaref = mne.preprocessing.ICA(n_components=None,max_iter=1000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mepo.info,meg=False,ref_meg=True)
        icaref.fit(mepo,picks=picks)
        #save the reference ica result in its own file
        icaref.save(proc_dir+sub+"_"+run+"_mepo-ref-ica.fif")
        #define the ica for MEG channels and fit it onto epoch file
        icameg = mne.preprocessing.ICA(n_components=None,max_iter=1000,method="picard") #parameters for ica on MEG channels
        picks = mne.pick_types(mepo.info,meg=True,ref_meg=False)
        icameg.fit(mepo,picks=picks)
        #save the MEG ica result in its own file
        icameg.save(proc_dir+sub+"_"+run+"_mepo-meg-ica.fif")
