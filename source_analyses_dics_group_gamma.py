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
freqs = {"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}
cycles = {"gamma_high":35}

# create lists for saving individual stcs to be averaged later on
all_diff_gamma_high = []
all_tonbas_gamma_high = []
# create lists to collect freq difference stc data arrays for permutation t test on source
X_gamma_high_diff = []
X_gamma_high_tonbas = []

for meg,mri in sub_dict.items():
    # load and prepare the STC data
    stc_fsavg_diff_gamma_high = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_gamma_high".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_tonbas_gamma_high = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_gamma_high".format(meg_dir,meg), subject='fsaverage')
    # collect the individual stcs into lists for averaging later
    all_diff_gamma_high.append(stc_fsavg_diff_gamma_high)
    X_gamma_high_diff.append(stc_fsavg_diff_gamma_high.data.T)
    all_tonbas_gamma_high.append(stc_fsavg_tonbas_gamma_high)
    X_gamma_high_tonbas.append(stc_fsavg_tonbas_gamma_high.data.T)

# create STC averages over all subjects for plotting
stc_gamma_high_sum = all_diff_gamma_high.pop()
for stc in all_diff_gamma_high:
    stc_gamma_high_sum = stc_gamma_high_sum + stc
NEM_all_stc_diff_gamma_high = stc_gamma_high_sum / len(sub_dict)

stc_gamma_high_ton_sum = all_tonbas_gamma_high.pop()
for stc in all_tonbas_gamma_high:
    stc_gamma_high_ton_sum = stc_gamma_high_ton_sum + stc
NEM_all_stc_tonbas_gamma_high = stc_gamma_high_ton_sum / len(sub_dict)

# plot difference N-P / Ton-Rest on fsaverage
NEM_all_stc_diff_gamma_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
NEM_all_stc_tonbas_gamma_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

# prepare source diff permutation t-tests
# there are two choices for clustering: normal cluster permutation with step-down p & TFCE with threshold-dict which is more robust (enhances connected peaks, but pushes disconnected ones down) -- choose one
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
connectivity = mne.spatial_src_connectivity(src)

X_gamma_high_diff = np.array(X_gamma_high_diff)
gh_t_obs, gh_clusters, gh_cluster_pv, gh_H0 = clu_gh = mne.stats.spatio_temporal_cluster_1samp_test(X_gamma_high_diff, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
# gh_t_obs, gh_clusters, gh_cluster_pv, gh_H0 = clu_gh = mne.stats.spatio_temporal_cluster_1samp_test(X_gamma_high_diff, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, t_power=1, out_type='indices')
gh_good_cluster_inds = np.where(gh_cluster_pv < 0.05)[0]
stc_gamma_high_clu_summ = mne.stats.summarize_clusters_stc(clu_gh, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
stc_gamma_high_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

X_gamma_high_tonbas = np.array(X_gamma_high_tonbas)
ght_t_obs, ght_clusters, ght_cluster_pv, ght_H0 = clu_ght = mne.stats.spatio_temporal_cluster_1samp_test(X_gamma_high_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
# ght_t_obs, ght_clusters, ght_cluster_pv, ght_H0 = clu_ght = mne.stats.spatio_temporal_cluster_1samp_test(X_gamma_high_tonbas, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, t_power=1, out_type='indices')
ght_good_cluster_inds = np.where(ght_cluster_pv < 0.05)[0]
stc_gamma_high_ton_clu_summ = mne.stats.summarize_clusters_stc(clu_ght, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
stc_gamma_high_ton_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

# plot difference N-P in plain t-values on fsaverage
NEM_all_stc_diff_gamma_high.data = gh_t_obs.T
NEM_all_stc_diff_gamma_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(2,4,6)})
NEM_all_stc_tonbas_gamma_high.data = ght_t_obs.T
NEM_all_stc_tonbas_gamma_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(2,4,6)})
