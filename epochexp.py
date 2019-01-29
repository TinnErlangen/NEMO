import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"

subjs = ["nc_NEM_12"]
runs = ["4"]

event_id = {'negative/r1': 110,'positive/r1': 120, 'negative/r2': 130, 'positive/r2': 140,
            'negative/s1': 150, 'positive/s1': 160,'negative/s2': 170, 'positive/s2': 180}

mini_epochs_num = 4
mini_epochs_len = 2

for sub in subjs:
    for run in runs:
        raw = mne.io.Raw(proc_dir+sub+'_'+run+'-raw.fif')
        events = list(np.load(proc_dir+sub+'_'+run+'_events.npy'))
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

        epochs = mne.Epochs(raw,new_events,event_id=event_id,baseline=None,tmin=0,tmax=mini_epochs_len,preload=True)
        #check epochs and labels
        print(epochs.event_id)
        print(epochs.events[:12])
        print(epochs[1:3])
        print(epochs['negative/r1'])
        epochs.save(proc_dir+sub+'_'+run+'-epo.fif')

        #look at them
        epochs.plot(n_epochs=8,n_channels=10)
        epochs.plot_psd(fmax=50)
        epochs.plot_psd_topomap()
