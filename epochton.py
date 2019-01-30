#step 2 B - creating epochs for tone baseline block

import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"

subjs = ["nc_NEM_12"]
runs = ["2"] # run 2 = tone baseline

#dictionary with conditions/triggers
event_id = {'ton_r1': 191,'ton_r2': 192, 'ton_s1': 193, 'ton_s2': 194}

mini_epochs_num = (5,4)
mini_epochs_len = 2

for sub in subjs:
    for run in runs:
        #loading raw data and original events
        raw = mne.io.Raw(proc_dir+sub+'_'+run+'-raw.fif')
        events = list(np.load(proc_dir+sub+'_'+run+'_events.npy'))
        #creating new event list with slices/starting time points for epochs
        new_events = []
        for e in events[:16]:
            for me in range(mini_epochs_num[0]):
                new_events.append(np.array(
                [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]]))
        for e in events[16:]:
            for me in range(mini_epochs_num[1]):
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
        print(epochs['ton_s1'])
        #saving to epoch file
        epochs.save(proc_dir+sub+'_'+run+'-epo.fif')

        #look at them (optional check)
        epochs.plot(n_epochs=8,n_channels=10)
        epochs.plot_psd(fmax=50)
        epochs.plot_psd_topomap()
