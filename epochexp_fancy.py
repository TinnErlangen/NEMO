import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"

subjs = ["nc_NEM_12"]
runs = ["3"]

event_id = {'110': 110, '111': 111, '112': 112, '113': 113, '120': 120, '121': 121, '122': 122,
            '123': 123, '130': 130, '131': 131, '132': 132, '133': 133, '140': 140, '141': 141,
            '142': 142, '143': 143, '150': 150, '151': 151, '152': 152, '153': 153, '160': 160,
            '161': 161, '162': 162, '163': 163, '170': 170, '171': 171, '172': 172, '173': 173,
            '180': 180, '181': 181, '182': 182, '183': 183, 'negative/r1': [110,111,112,113],
            'negative/r2': [130,131,132,133]}

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
                    [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]+me]))
        new_events = np.array(new_events).astype(int)
        print(new_events[:25,:])
        print(len(new_events))
        print(np.unique(new_events[:,2]))
        epochs = mne.Epochs(raw,new_events,event_id=event_id,baseline=None,tmin=0,tmax=mini_epochs_len,preload=True)
