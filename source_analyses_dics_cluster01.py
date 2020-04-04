import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_26":"ENR41"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

# prepare for source diff permutation t-test
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
connectivity = mne.spatial_src_connectivity(src)
threshold = 2.845

for freq,vals in freqs.items():

    print("Running analyses for '{}'\n".format(freq))
    # list for collecting stcs for group average for plotting
    all_diff = []
    # list for data arrays for permutation t-test on source
    X_diff = []

    for meg,mri in sub_dict.items():
        # load and prepare the STC data
        stc_fsavg_diff = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_{}".format(meg_dir,meg,freq), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
        # collect the individual stcs into lists for averaging later
        all_diff.append(stc_fsavg_diff)
        X_diff.append(stc_fsavg_diff.data.T)

    # create STC average over all subjects for plotting
    stc_sum = all_diff.pop()
    for stc in all_diff:
        stc_sum = stc_sum + stc
    NEM_all_stc_diff = stc_sum / len(sub_dict)
    # plot difference N-P on fsaverage
    NEM_all_stc_diff.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)

    # now do cluster permutation analysis
    X_diff = np.array(X_diff)
    t_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(X_diff, n_permutations=1024, threshold = threshold, tail=0, connectivity=connectivity, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
    # t_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(X_diff, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=4, t_power=1, out_type='indices')

    # plot difference N-P in plain t-values on fsaverage
    NEM_all_stc_diff.data = t_obs.T
    NEM_all_stc_diff.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(2,4,6)})

    # get significant clusters and plot
    good_cluster_inds = np.where(cluster_pv < 0.05)[0]
    if len(good_cluster_inds):
        stc_clu_summ = mne.stats.summarize_clusters_stc(clu, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=None)
        stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')
    else:
        print("No sign. clusters found")
