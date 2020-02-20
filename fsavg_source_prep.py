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
# fsavg_src.save(meg_dir+"fsaverage_oct6-src.fif", overwrite=True)
# fsavg_src.save(meg_dir+"fsaverage_ico5-src.fif", overwrite=True)

bem_model = mne.make_bem_model('fsaverage', subjects_dir=mri_dir, conductivity=[0.3])
bem = mne.make_bem_solution(bem_model)
mne.write_bem_solution(meg_dir+"fsaverage-bem.fif",bem)
mne.viz.plot_bem(subject='fsaverage', subjects_dir=mri_dir,
             brain_surfaces='white', src=fsavg_src, orientation='coronal')

# create a mixed source space from here, adding subcortical volumes from label + aseg.mgz
labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
              'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']
fsavg_limb_src = mne.setup_volume_source_space('fsaverage', mri=mri_dir+"/fsaverage/mri/aseg.mgz", pos=5.0, bem=bem,
                                        volume_label=labels_limb, subjects_dir=mri_dir,add_interpolator=True,verbose=True)

# Generate the mixed source space
fsavg_src += fsavg_limb_src
# Visualize the source spaces
fsavg_limb_src.plot(subjects_dir=mri_dir)
fsavg_src.plot(subjects_dir=mri_dir)
# print out the number of spaces and points
n = sum(fsavg_src[i]['nuse'] for i in range(len(fsavg_src)))
print('the fsavg_src space contains %d spaces and %d points' % (len(fsavg_src), n))
# save the mixed source space
fsavg_src.save(meg_dir+"fsaverage_ico5_limb_mix-src.fif", overwrite=True)
