# spatio-temporal cluster test over frequencies 1-90 Hz

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
import random

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

# PREPARE : create CSD and STC files for all subjects (group analyses will be done without exluded ones - see below)

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_14":"FIN23"}
# # sub_dict = {"NEM_15":"KIL13","NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48",
#               "NEM_21":"WKI71_fa","NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
#               "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72","NEM_30":"DIU11","NEM_31":"BLE94",
#               "NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa",
#               "NEM_37":"EAM67"} # these are waiting for the next batch
# # sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11",} # these are done

for meg,mri in sub_dict.items():
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
    neg = epo_exp['negative']
    pos = epo_exp['positive']
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = ton.info['dev_head_t']
    epo_bas = mne.concatenate_epochs([rest,ton])
    # load the forward models from each experimental block
    fwd_rest = mne.read_forward_solution("{dir}nc_{meg}_1-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_ton =  mne.read_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_base = mne.read_forward_solution("{dir}nc_{meg}_base-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_exp = mne.read_forward_solution("{dir}nc_{meg}_exp-fwd.fif".format(dir=meg_dir,meg=meg))
    # make and save CSDs for 1-90 Hz range
    frequencies = np.linspace(1,90,num=90)
    csd_exp = csd_morlet(epo_exp, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_exp.save("{dir}nc_{meg}-csd_exp_1-90.h5".format(dir=meg_dir,meg=meg))
    csd_bas = csd_morlet(epo_bas, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_bas.save("{dir}nc_{meg}-csd_bas_1-90.h5".format(dir=meg_dir,meg=meg))
    csd_ton = csd_morlet(ton, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_ton.save("{dir}nc_{meg}-csd_ton_1-90.h5".format(dir=meg_dir,meg=meg))
    csd_rest = csd_morlet(rest, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_rest.save("{dir}nc_{meg}-csd_rest_1-90.h5".format(dir=meg_dir,meg=meg))
    csd_neg = csd_morlet(neg, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_neg.save("{dir}nc_{meg}-csd_neg_1-90.h5".format(dir=meg_dir,meg=meg))
    csd_pos = csd_morlet(pos, frequencies=frequencies, n_jobs=8, n_cycles=7, decim=1)
    csd_pos.save("{dir}nc_{meg}-csd_pos_1-90.h5".format(dir=meg_dir,meg=meg))
    # make DICS filters for exp and baseline
    filters_exp = make_dics(epo_exp.info,fwd_exp,csd_exp,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas = make_dics(epo_bas.info,fwd_base,csd_bas,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # apply the DICS beamformers to get source Estimates for (base) rest, tonbas & (exp) ton, neg, pos using common filters & save to file
    stc_rest, freqs_rest = apply_dics_csd(csd_rest,filters_bas)
    stc_rest.save(fname=meg_dir+"nc_{}_stc_rest_1-90".format(meg))
    stc_tonbas, freqs_tonbas = apply_dics_csd(csd_ton,filters_bas)
    stc_tonbas.save(fname=meg_dir+"nc_{}_stc_tonbas_1-90".format(meg))
    stc_ton, freqs_ton = apply_dics_csd(csd_ton,filters_exp)
    stc_ton.save(fname=meg_dir+"nc_{}_stc_ton_1-90".format(meg))
    stc_neg, freqs_neg = apply_dics_csd(csd_neg,filters_exp)
    stc_neg.save(fname=meg_dir+"nc_{}_stc_neg_1-90".format(meg))
    stc_pos, freqs_pos = apply_dics_csd(csd_pos,filters_exp)
    stc_pos.save(fname=meg_dir+"nc_{}_stc_pos_1-90".format(meg))
    # calculate the difference between conditions div. by baseline & save to file (for base and for exp)
    stc_diff_tonbas = (stc_tonbas - stc_rest) / stc_rest
    stc_diff_tonbas.save(fname=meg_dir+"nc_{}_stc_diff_tonbas_1-90".format(meg))
    stc_diff_emo = (stc_neg - stc_pos) / stc_ton
    stc_diff_emo.save(fname=meg_dir+"nc_{}_stc_diff_emo_1-90".format(meg))
    # morph the resulting stcs to fsaverage & save  (to be loaded again and averaged)
    #src = mne.read_source_spaces("{}nc_{}-src.fif".format(meg_dir,meg))  ## as no. of vertices doesn't match between src and stc (got reduced by building forward model with mindist 5.0) - stc is used directly for morphing
    morph_bas = mne.compute_source_morph(stc_diff_tonbas,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    stc_fsavg_diff_tonbas = morph_bas.apply(stc_diff_tonbas)
    stc_fsavg_diff_tonbas.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_tonbas_1-90".format(meg))
    morph_exp = mne.compute_source_morph(stc_diff_emo,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    stc_fsavg_diff_emo = morph_exp.apply(stc_diff_emo)
    stc_fsavg_diff_emo.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_emo_1-90".format(meg))

# # now do GROUP ANALYSES - with final subject sample
#
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_17":"DEN59","NEM_18":"SAG13",
#            "NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
#            "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72",
#            "NEM_34":"KER27","NEM_36":"BRA52_fa","NEM_16":"KIO12","NEM_20":"PAG48","NEM_31":"BLE94","NEM_35":"MUN79"}
# excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# # sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
#
# # create lists for saving individual stcs to be averaged later on
# all_diff_tonbas=[]
# all_diff_emo=[]
# # create lists to collect freq difference stc data arrays for permutation t test on source
# X_tonbas_diff = []
# X_emo_diff = []
#
# for meg,mri in sub_dict.items():
#     # load and prepare the STC data
#     stc_fsavg_diff_tonbas = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_tonbas_1-90".format(meg_dir,meg), subject='fsaverage')  ## works without file ending like this (loads both lh and rh)
#     stc_fsavg_diff_emo = mne.read_source_estimate("{}nc_{}_stc_fsavg_diff_emo_1-90".format(meg_dir,meg), subject='fsaverage')
#     # collect the individual stcs into lists for averaging later
#     all_diff_tonbas.append(stc_fsavg_diff_tonbas)
#     X_tonbas_diff.append(stc_fsavg_diff_tonbas.data.T)
#     all_diff_emo.append(stc_fsavg_diff_emo)
#     X_emo_diff.append(stc_fsavg_diff_emo.data.T)
#
# # create STC averages over all subjects for plotting - tonbas & emo
# stc_tonbas_sum = all_diff_tonbas.pop()
# for stc in all_diff_tonbas:
#     stc_tonbas_sum = stc_tonbas_sum + stc
# NEM_all_stc_diff_tonbas = stc_tonbas_sum / len(sub_dict)
#
# stc_emo_sum = all_diff_emo.pop()
# for stc in all_diff_emo:
#     stc_emo_sum = stc_emo_sum + stc
# NEM_all_stc_diff_emo = stc_emo_sum / len(sub_dict)
#
# # plot differences tonbas-rest and Neg-Pos on fsaverage
# NEM_all_stc_diff_tonbas.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
# NEM_all_stc_diff_emo.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True)
#
# # prepare source diff permutation t-test
# src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
# connectivity = mne.spatial_src_connectivity(src)
#
# # do permutation t-test and plot it for tonbas_diff
# X_tonbas_diff = np.array(X_tonbas_diff)
# ton_t_obs, ton_clusters, ton_cluster_pv, ton_H0 = clu_ton = mne.stats.spatio_temporal_cluster_1samp_test(X_tonbas_diff, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=8, step_down_p=0.05, t_power=1, out_type='indices')
# # ton_t_obs, ton_clusters, ton_cluster_pv, ton_H0 = clu_ton = mne.stats.spatio_temporal_cluster_1samp_test(X_tonbas_diff, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=8, t_power=1, out_type='indices')
# ton_good_cluster_inds = np.where(ton_cluster_pv < 0.05)[0]
# stc_tonbas_clu_summ = mne.stats.summarize_clusters_stc(clu_ton, p_thresh=0.05, tstep=1, tmin=1, subject='fsaverage', vertices=None) # see if have to adjust time parameters for freqs
# stc_tonbas_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')
#
# # do permutation t-test and plot it for emo_diff
# X_emo_diff = np.array(X_emo_diff)
# emo_t_obs, emo_clusters, emo_cluster_pv, emo_H0 = clu_emo = mne.stats.spatio_temporal_cluster_1samp_test(X_emo_diff, n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=8, step_down_p=0.05, t_power=1, out_type='indices')
# # emo_t_obs, emo_clusters, emo_cluster_pv, emo_H0 = clu_emo = mne.stats.spatio_temporal_cluster_1samp_test(X_emo_diff, threshold = dict(start=0,step=0.2), n_permutations=1024, tail=0, connectivity=connectivity, n_jobs=8, t_power=1, out_type='indices')
# emo_good_cluster_inds = np.where(emo_cluster_pv < 0.05)[0]
# stc_emo_clu_summ = mne.stats.summarize_clusters_stc(clu_emo, p_thresh=0.05, tstep=1, tmin=1, subject='fsaverage', vertices=None) # see if have to adjust time parameters for freqs
# stc_emo_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,spacing='ico5')
