import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt

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
freqs = {"theta":list(np.arange(4,7)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

#
# for meg,mri in sub_dict.items():
#     # load and prepare the MEG data
#     ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
#     epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
#     # load the forward models from each experimental block
#     fwd_ton =  mne.read_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg))
#     fwd_exp = mne.read_forward_solution("{dir}nc_{meg}_exp-fwd.fif".format(dir=meg_dir,meg=meg))
#     for freq,vals in freqs.items():
#         # load the csd matrices needed: i.e. for every freq band, get exp, ton, neg and pos (and maybe bas & rest)
#         csd_exp = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_{freq}.h5".format(dir=meg_dir,meg=meg,freq=freq))
#         csd_ton = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_{freq}.h5".format(dir=meg_dir,meg=meg,freq=freq))
#         csd_neg = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_{freq}.h5".format(dir=meg_dir,meg=meg,freq=freq))
#         csd_pos = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_{freq}.h5".format(dir=meg_dir,meg=meg,freq=freq))
#         # make DICS beamformers / filters (common filters) for different freq bands
#         filters_exp = make_dics(epo_exp.info,fwd_exp,csd_exp,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
#         # apply the DICS beamformers to get source Estimates for ton, neg, pos using common filters & save to file
#         stc_ton, freqs_ton = apply_dics_csd(csd_ton,filters_exp)
#         stc_neg, freqs_neg = apply_dics_csd(csd_neg,filters_exp)
#         stc_pos, freqs_pos = apply_dics_csd(csd_pos,filters_exp)
#         # calculate the difference between conditions div. by baseline & save to file
#         stc_diff = (stc_neg - stc_pos) / stc_ton
#         # morph the resulting stcs to fsaverage & save  (to be loaded again and averaged)
#         morph = mne.compute_source_morph(stc_diff,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
#         stc_fsavg_diff = morph.apply(stc_diff)
#         stc_fsavg_diff.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_F_{}".format(meg,freq))


# prepare for source diff permutation t-test
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
connectivity = mne.spatial_src_connectivity(src)
threshold = 2.086

# DO GROUP PLOTS AND CLUSTER PERMUTATIONS
save_dir = "D:/NEMO_analyses/plots/exp_vs_ton/"
cond = "neg_pos"
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
poslims = {"theta":(1,2,3),"alpha":(2,4,6),"beta_low":(3,5,7),"beta_high":(4,6,9),"gamma":(8,14,20),"gamma_high":(8,14,20)}

for freq,vals in freqs.items():

    print("Running analyses for '{}'\n".format(freq))
    # list for collecting stcs for group average for plotting
    all_diff = []
    # list for data arrays for permutation t-test on source
    X_diff = []

    for meg,mri in sub_dict.items():
        # load and prepare the STC data
        stc_fsavg_diff = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_F_{}".format(meg_dir,meg,freq), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
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
        stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5',colormap='coolwarm',clim={'kind':'value','pos_lims': poslims[freq]})
        fig = mlab.figure(size=(300, 300))
        brain = stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',spacing='ico5',colormap='coolwarm',clim={'kind':'value','pos_lims': poslims[freq]},figure=fig)
        brain.add_annotation('HCPMMP1_combined', borders=1, alpha=0.9)
        mlab.view(0, 90, 450, [0, 0, 0])
        mlab.savefig('{d}{c}_diff_{f}_clu_F_rh.png'.format(d=save_dir,c=cond,f=freq), magnification=4)
        mlab.view(180, 90, 450, [0, 0, 0])
        mlab.savefig('{d}{c}_diff_{f}_clu_F_lh.png'.format(d=save_dir,c=cond,f=freq), magnification=4)
        mlab.view(180, 0, 450, [0, 10, 0])
        mlab.savefig('{d}{c}_diff_{f}_clu_F_top.png'.format(d=save_dir,c=cond,f=freq), magnification=4)
        mlab.view(180, 180, 480, [0, 10, 0])
        mlab.savefig('{d}{c}_diff_{f}_clu_F_bottom.png'.format(d=save_dir,c=cond,f=freq), magnification=4)
        mlab.close(fig)
    else:
        print("No sign. clusters found")
