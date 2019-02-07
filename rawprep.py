#first step - code for reading events, filtering and resampling
import numpy as np
import mne

#file locations
base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
raw_dir = base_dir+"raw/"
#location for processed data files
proc_dir = base_dir+"proc/"

subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]
#runs = ["1"]

notches = [16.7, 24, 50, 62, 100, 150, 200]
breadths = np.array([0.25, 2.0, 1.5, 0.5, 0.5, 0.5, 0.5])

for sub in subjs:
    sub_path = raw_dir+sub+"/"
    for run in runs:
        run_path = sub_path+run+"/"
        #initial reading of 4D raw data
        rawmeg = mne.io.read_raw_bti(run_path+"c,rfhp1.0Hz",preload=True,rename_channels=False)
        #finding events, adjusting 4D trigger error
        rawmeg_events = mne.find_events(rawmeg,stim_channel="TRIGGER",initial_event=True,consecutive=True)
        for i_idx in range(len(rawmeg_events)):
            if rawmeg_events[i_idx,2]>4000:
                rawmeg_events[i_idx,2]=rawmeg_events[i_idx,2]-4095
        #filtering
        rawmeg.notch_filter(notches,n_jobs="cuda",notch_widths=breadths) #cuda is for use of GPU for processing, if not available on your machine, leave out
        #resampling - maintaining original event time points (!)
        rawmeg,rawmeg_events = rawmeg.resample(200,events=rawmeg_events,n_jobs="cuda") #see above for cuda
        meg_events = rawmeg_events
        #saving event file & new raw file
        np.save(proc_dir+sub+"_"+run+"_events.npy",meg_events)
        rawmeg.save(proc_dir+sub+"_"+run+"-raw.fif",overwrite=True)
