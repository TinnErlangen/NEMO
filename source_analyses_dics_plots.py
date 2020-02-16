import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48",
           "NEM_21":"WKI71_fa","NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72","NEM_30":"DIU11","NEM_31":"BLE94",
           "NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa",
           "NEM_37":"EAM67"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
#            "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48"}
sub_dict = {"NEM_10":"GIZ04"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}

for meg,mri in sub_dict.items():
    # load and prepare the STC data
    # original difference neg-pos/ton
    stc_diff_alpha = mne.read_source_estimate("{}nc_{}_stc_diff_alpha".format(meg_dir,meg), subject=mri)  ## works without file ending like this (loads both lh and rh)
    stc_diff_theta = mne.read_source_estimate("{}nc_{}_stc_diff_theta".format(meg_dir,meg), subject=mri)
    stc_diff_beta_low = mne.read_source_estimate("{}nc_{}_stc_diff_beta_low".format(meg_dir,meg), subject=mri)
    stc_diff_beta_high = mne.read_source_estimate("{}nc_{}_stc_diff_beta_high".format(meg_dir,meg), subject=mri)
    stc_diff_gamma = mne.read_source_estimate("{}nc_{}_stc_diff_gamma".format(meg_dir,meg), subject=mri)
    # morphed to fsaverage
    stc_fsavg_diff_alpha = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_alpha".format(meg_dir,meg), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
    stc_fsavg_diff_theta = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_theta".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_beta_low = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_beta_low".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_beta_high = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_beta_high".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_gamma = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_gamma".format(meg_dir,meg), subject='fsaverage')


    # plot individual difference stcs (for each freq band)  & save into 2x5 figure showing both hemispheres per freq band
    stc_diff_alpha.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.04,0.17,0.3)})
    stc_diff_theta.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.035,0.1425,0.25)})
    stc_diff_beta_low.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.03,0.115,0.2)})
    stc_diff_beta_high.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.02,0.085,0.15)})
    stc_diff_gamma.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.015,0.0575,0.1)})

    # plot individual difference stcs (for each freq band) morphed on fsaverage brain  & save into 2x5 figure showing both hemispheres per freq band
    stc_fsavg_diff_alpha.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.04,0.17,0.3)})
    stc_fsavg_diff_theta.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.035,0.1425,0.25)})
    stc_fsavg_diff_beta_low.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.03,0.115,0.2)})
    stc_fsavg_diff_beta_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.02,0.085,0.15)})
    stc_fsavg_diff_gamma.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0.015,0.0575,0.1)})
