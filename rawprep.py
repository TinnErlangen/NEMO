#first step - code for reading events, filtering and resampling
import numpy as np
import mne

#file locations
base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
raw_dir = base_dir+"raw/"
#location for processed data files
proc_dir = base_dir+"proc/"

subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_25"]
runs = ["1","2","3","4"]
#runs = ["4"]

#notches = [16.7, 24, 50, 62, 100, 150, 200]
notches = [50, 62, 100, 150, 200]
#breadths = np.array([0.25, 2.0, 1.5, 0.5, 0.5, 0.5, 0.5])
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])

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
        #filtering - also on ref channels (for combined ICA later)
        picks = mne.pick_types(rawmeg.info, meg=True,ref_meg=True)
        rawmeg.notch_filter(notches,picks=picks,n_jobs="cuda",notch_widths=breadths) #cuda is for use of GPU for processing, if not available on your machine, leave out
        #resampling - maintaining original event time points (!)
        rawmeg,rawmeg_events = rawmeg.resample(200,events=rawmeg_events,n_jobs="cuda") #see above for cuda
        meg_events = rawmeg_events
        #saving event file & new raw file
        np.save(proc_dir+sub+"_"+run+"_events.npy",meg_events)
        rawmeg.save(proc_dir+sub+"_"+run+"-raw.fif",overwrite=True)
