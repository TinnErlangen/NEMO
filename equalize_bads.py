import mne
import numpy as np

proc_dir = "D:/NEMO_analyses/proc/"
subjs = ["nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
# subjs = ["nc_NEM_10"]
#runs = ["1","2","3","4"]

for sub in subjs:
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}{sub}_1_ica-epo.fif".format(dir=proc_dir,sub=sub))
    ton = mne.read_epochs("{dir}{sub}_2_ica-epo.fif".format(dir=proc_dir,sub=sub))
    epo_a = mne.read_epochs("{dir}{sub}_3_ica-epo.fif".format(dir=proc_dir,sub=sub))
    epo_b = mne.read_epochs("{dir}{sub}_4_ica-epo.fif".format(dir=proc_dir,sub=sub))
    # read bad channels and append to common list
    bads = rest.info['bads']
    for i in ton.info['bads']:
        if i not in bads:
            bads.append(i)
    for i in epo_a.info['bads']:
        if i not in bads:
            bads.append(i)
    for i in epo_b.info['bads']:
        if i not in bads:
            bads.append(i)
    # apply bads to all and save
    rest.info['bads'] = bads
    ton.info['bads'] = bads
    epo_a.info['bads'] = bads
    epo_b.info['bads'] = bads
    rest.save("{dir}{sub}_1_ica-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    ton.save("{dir}{sub}_2_ica-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    epo_a.save("{dir}{sub}_3_ica-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    epo_b.save("{dir}{sub}_4_ica-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
