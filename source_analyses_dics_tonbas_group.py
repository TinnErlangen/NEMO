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
all_tonbas_alpha=[]
all_tonbas_theta=[]
all_tonbas_beta_low=[]
all_tonbas_beta_high=[]
all_tonbas_gamma=[]

# create lists to collect freq difference stc data arrays for permutation t test on source
X_alpha_tonbas = []
X_theta_tonbas = []
X_beta_low_tonbas = []
X_beta_high_tonbas = []
X_gamma_tonbas = []

for meg,mri in sub_dict.items():
    # load and prepare the STC data
    stc_fsavg_tonbas_alpha = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_alpha".format(meg_dir,meg), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
    stc_fsavg_tonbas_theta = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_theta".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_tonbas_beta_low = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_beta_low".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_tonbas_beta_high = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_beta_high".format(meg_dir,meg), subject='fsaverage')
    stc_fsavg_tonbas_gamma = mne.read_source_estimate("{}nc_{}_stc_fsavg_tonbas_gamma".format(meg_dir,meg), subject='fsaverage')
    # collect the individual stcs into lists for averaging later
    all_tonbas_alpha.append(stc_fsavg_tonbas_alpha)
    X_alpha_tonbas.append(stc_fsavg_tonbas_alpha.data.T)
    all_tonbas_theta.append(stc_fsavg_tonbas_theta)
    X_theta_tonbas.append(stc_fsavg_tonbas_theta.data.T)
    all_tonbas_beta_low.append(stc_fsavg_tonbas_beta_low)
    X_beta_low_tonbas.append(stc_fsavg_tonbas_beta_low.data.T)
    all_tonbas_beta_high.append(stc_fsavg_tonbas_beta_high)
    X_beta_high_tonbas.append(stc_fsavg_tonbas_beta_high.data.T)
    all_tonbas_gamma.append(stc_fsavg_tonbas_gamma)
    X_gamma_tonbas.append(stc_fsavg_tonbas_gamma.data.T)

# create STC averages over all subjects for plotting
stc_alpha_sum = all_tonbas_alpha.pop()
for stc in all_tonbas_alpha:
    stc_alpha_sum = stc_alpha_sum + stc
NEM_all_stc_tonbas_alpha = stc_alpha_sum / len(sub_dict)

stc_theta_sum = all_tonbas_theta.pop()
for stc in all_tonbas_theta:
    stc_theta_sum = stc_theta_sum + stc
NEM_all_stc_tonbas_theta = stc_theta_sum / len(sub_dict)

stc_beta_low_sum = all_tonbas_beta_low.pop()
for stc in all_tonbas_beta_low:
    stc_beta_low_sum = stc_beta_low_sum + stc
NEM_all_stc_tonbas_beta_low = stc_beta_low_sum / len(sub_dict)

stc_beta_high_sum = all_tonbas_beta_high.pop()
for stc in all_tonbas_beta_high:
    stc_beta_high_sum = stc_beta_high_sum + stc
NEM_all_stc_tonbas_beta_high = stc_beta_high_sum / len(sub_dict)

stc_gamma_sum = all_tonbas_gamma.pop()
for stc in all_tonbas_gamma:
    stc_gamma_sum = stc_gamma_sum + stc
NEM_all_stc_tonbas_gamma = stc_gamma_sum / len(sub_dict)

# plot difference N-P on fsaverage
# NEM_all_stc_tonbas_alpha.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
# NEM_all_stc_tonbas_theta.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
# NEM_all_stc_tonbas_beta_low.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
# NEM_all_stc_tonbas_beta_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
# NEM_all_stc_tonbas_gamma.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

# prepare source alpha tonbas permutation t-test
# there are two choices for clustering: normal cluster permutation with step-down p & TFCE with threshold-dict which is more robust (enhances connected peaks, but pushes disconnected ones down) -- choose one
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
connectivity = mne.spatial_src_connectivity(src)

X_alpha_tonbas = np.array(X_alpha_tonbas)
a_t_obs, a_clusters, a_cluster_pv, a_H0 = clu_a = mne.stats.spatio_temporal_cluster_1samp_test(X_alpha_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
# a_t_obs, a_clusters, a_cluster_pv, a_H0 = clu_a = mne.stats.spatio_temporal_cluster_1samp_test(X_alpha_tonbas, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, t_power=1, out_type='indices')
# get significant clusters and plot
a_good_cluster_inds = np.where(a_cluster_pv < 0.05)[0]
# stc_alpha_clu_summ = mne.stats.summarize_clusters_stc(clu_a, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
# stc_alpha_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

# do permutation t-test and plot it for all other freq bands
X_theta_tonbas = np.array(X_theta_tonbas)
th_t_obs, th_clusters, th_cluster_pv, th_H0 = clu_th = mne.stats.spatio_temporal_cluster_1samp_test(X_theta_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
th_good_cluster_inds = np.where(th_cluster_pv < 0.05)[0]
# stc_theta_clu_summ = mne.stats.summarize_clusters_stc(clu_th, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
# stc_theta_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

X_beta_low_tonbas = np.array(X_beta_low_tonbas)
bl_t_obs, bl_clusters, bl_cluster_pv, bl_H0 = clu_bl = mne.stats.spatio_temporal_cluster_1samp_test(X_beta_low_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
bl_good_cluster_inds = np.where(bl_cluster_pv < 0.05)[0]
# stc_beta_low_clu_summ = mne.stats.summarize_clusters_stc(clu_bl, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
# stc_beta_low_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

X_beta_high_tonbas = np.array(X_beta_high_tonbas)
bh_t_obs, bh_clusters, bh_cluster_pv, bh_H0 = clu_bh = mne.stats.spatio_temporal_cluster_1samp_test(X_beta_high_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
bh_good_cluster_inds = np.where(bh_cluster_pv < 0.05)[0]
# stc_beta_high_clu_summ = mne.stats.summarize_clusters_stc(clu_bh, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
# stc_beta_high_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

X_gamma_tonbas = np.array(X_gamma_tonbas)
g_t_obs, g_clusters, g_cluster_pv, g_H0 = clu_g = mne.stats.spatio_temporal_cluster_1samp_test(X_gamma_tonbas, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
g_good_cluster_inds = np.where(g_cluster_pv < 0.05)[0]
# stc_gamma_clu_summ = mne.stats.summarize_clusters_stc(clu_g, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
# stc_gamma_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')

# plot difference N-P in plain t-values on fsaverage
NEM_all_stc_tonbas_alpha.data = a_t_obs.T
NEM_all_stc_tonbas_alpha.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(3,4.5,6)})
NEM_all_stc_tonbas_theta.data = th_t_obs.T
NEM_all_stc_tonbas_theta.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(3,4.5,6)})
NEM_all_stc_tonbas_beta_low.data = bl_t_obs.T
NEM_all_stc_tonbas_beta_low.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(3,4.5,6)})
NEM_all_stc_tonbas_beta_high.data = bh_t_obs.T
NEM_all_stc_tonbas_beta_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(3,4.5,6)})
NEM_all_stc_tonbas_gamma.data = g_t_obs.T
NEM_all_stc_tonbas_gamma.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(3,4.5,6)})
