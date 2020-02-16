## preparation of source space and forward solutions

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"

fsavg_src = mne.setup_source_space("fsaverage",surface="white",subjects_dir=mri_dir,spacing='ico5',n_jobs=4)  ## uses 'oct6' as default, i.e. 4.9mm spacing appr.
#fsavg_src.save(meg_dir+"fsaverage_oct6-src.fif", overwrite=True)
fsavg_src.save(meg_dir+"fsaverage_ico5-src.fif", overwrite=True)
