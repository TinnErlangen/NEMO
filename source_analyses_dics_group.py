import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_17":"DEN59","NEM_18":"SAG13",
           "NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72",
           "NEM_34":"KER27","NEM_36":"BRA52_fa","NEM_16":"KIO12","NEM_20":"PAG48","NEM_31":"BLE94","NEM_35":"MUN79"}
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
#            "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48"}
# sub_dict = {"NEM_26":"ENR41"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}

# create lists for saving individual stcs to be averaged later on
all_diff_alpha=[]
all_diff_theta=[]
all_diff_beta_low=[]
all_diff_beta_high=[]
all_diff_gamma=[]

for meg,mri in sub_dict.items():
    # load and prepare the STC data
    stc_fsavg_diff_alpha = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_alpha".format(meg_dir,meg), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
    stc_fsavg_diff_theta = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_theta".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_beta_low = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_beta_low".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_beta_high = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_beta_high".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_diff_gamma = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_gamma".format(meg_dir,meg), subject='fsaverage')
    # collect the individual stcs into lists for averaging later
    all_diff_alpha.append(stc_fsavg_diff_alpha)
    all_diff_theta.append(stc_fsavg_diff_theta)
    all_diff_beta_low.append(stc_fsavg_diff_beta_low)
    all_diff_beta_high.append(stc_fsavg_diff_beta_high)
    all_diff_gamma.append(stc_fsavg_diff_gamma)

stc_alpha_sum = all_diff_alpha.pop()
for stc in all_diff_alpha:
    stc_alpha_sum = stc_alpha_sum + stc
NEM_all_stc_diff_alpha = stc_alpha_sum / len(sub_dict)
NEM_all_stc_diff_alpha.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

stc_theta_sum = all_diff_theta.pop()
for stc in all_diff_theta:
    stc_theta_sum = stc_theta_sum + stc
NEM_all_stc_diff_theta = stc_theta_sum / len(sub_dict)
NEM_all_stc_diff_theta.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

stc_beta_low_sum = all_diff_beta_low.pop()
for stc in all_diff_beta_low:
    stc_beta_low_sum = stc_beta_low_sum + stc
NEM_all_stc_diff_beta_low = stc_beta_low_sum / len(sub_dict)
NEM_all_stc_diff_beta_low.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

stc_beta_high_sum = all_diff_beta_high.pop()
for stc in all_diff_beta_high:
    stc_beta_high_sum = stc_beta_high_sum + stc
NEM_all_stc_diff_beta_high = stc_beta_high_sum / len(sub_dict)
NEM_all_stc_diff_beta_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

stc_gamma_sum = all_diff_gamma.pop()
for stc in all_diff_gamma:
    stc_gamma_sum = stc_gamma_sum + stc
NEM_all_stc_diff_gamma = stc_gamma_sum / len(sub_dict)
NEM_all_stc_diff_gamma.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

#  get label lists for peaks
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))

thresh_a = 0.05
stc_NEM_all_diff_alpha_peak = NEM_all_stc_diff_alpha.copy()
stc_NEM_all_diff_alpha_peak.data[np.abs(stc_NEM_all_diff_alpha_peak.data) < thresh_a] = 0
act_labels_alpha = mne.stc_to_label(stc_NEM_all_diff_alpha_peak,src=src,connected=True,subjects_dir=mri_dir)

thresh_t = 0.05
stc_NEM_all_diff_alpha_peak = NEM_all_stc_diff_alpha.copy()
stc_NEM_all_diff_alpha_peak.data[np.abs(stc_NEM_all_diff_alpha_peak.data) < thresh_t] = 0
act_labels_alpha = mne.stc_to_label(stc_NEM_all_diff_alpha_peak,src=src,connected=True,subjects_dir=mri_dir)
