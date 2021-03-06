#step 2 C - creating epochs for experiment blocks

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
runs = ["3","4"] # runs 3,4 = blocks A+B of experiment

#dictionary with conditions/triggers
event_id = {'negative/r1': 110,'positive/r1': 120, 'negative/r2': 130, 'positive/r2': 140,
            'negative/s1': 150, 'positive/s1': 160,'negative/s2': 170, 'positive/s2': 180}

mini_epochs_num = 4
mini_epochs_len = 2

for sub in subjs:
    for run in runs:
        #loading raw data and original events
        raw = mne.io.Raw(proc_dir+sub+'_'+run+'-raw.fif')
        events = list(np.load(proc_dir+sub+'_'+run+'_events.npy'))
        #creating new event list with slices/starting time points for epochs
        new_events = []
        for e in events:
            if e[2] > 100:
                for me in range(mini_epochs_num):
                    new_events.append(np.array(
                    [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]]))
        new_events = np.array(new_events).astype(int)
        #check if events alright
        print(new_events[:25,:])
        print(len(new_events))
        print(np.unique(new_events[:,2]))

        #creating Epoch object from new event list
        epochs = mne.Epochs(raw,new_events,event_id=event_id,baseline=None,tmin=0,tmax=mini_epochs_len,preload=True)
        #check epochs and labels
        print(epochs.event_id)
        print(epochs.events[:12])
        print(epochs[1:3])
        print(epochs['negative/r1'])
        #saving to epoch file
        epochs.save(proc_dir+sub+'_'+run+'-epo.fif')

        #look at them (optional check)
        # epochs.plot(n_epochs=8,n_channels=32)
        # epochs.plot_psd(fmax=50,average=False)
        #epochs.plot_psd_topomap()
