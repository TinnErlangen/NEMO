#step 4 - do ICA on annotated epoch files

import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_25"]
runs = ["1","2","3","4"]
#runs = ["1","2"]

for sub in subjs:
    for run in runs:
        #load the annotated epoch file
        mepo = mne.read_epochs(proc_dir+sub+"_"+run+"_m-epo.fif")
        mepo.info["bads"] += ["MRyA","MRyaA"] # add broken reference channels to bad channels list

        #define the ica for reference channels and fit it onto epoch file
        icaref = mne.preprocessing.ICA(n_components=6,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mepo.info,meg=False,ref_meg=True)
        icaref.fit(mepo,picks=picks)
        #save the reference ica result in its own file
        icaref.save(proc_dir+sub+"_"+run+"_mepo-ref-ica.fif")

        #define the ica for MEG channels and fit it onto epoch file
        icameg = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard") #parameters for ica on MEG channels
        picks = mne.pick_types(mepo.info,meg=True,ref_meg=False)
        icameg.fit(mepo,picks=picks)
        #save the MEG ica result in its own file
        icameg.save(proc_dir+sub+"_"+run+"_mepo-meg-ica.fif")

        icaall = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mepo.info,meg=True,ref_meg=True)
        icaall.fit(mepo,picks=picks)
        icaall.save(proc_dir+sub+"_"+run+"_mepo-ica.fif")
