import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

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
#sub_dict = {"NEM_26":"ENR41"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}

for meg,mri in sub_dict.items():
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
    # load the forward models from each experimental block
    fwd_rest = mne.read_forward_solution("{dir}nc_{meg}_1-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_ton =  mne.read_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_base = mne.read_forward_solution("{dir}nc_{meg}_base-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_exp = mne.read_forward_solution("{dir}nc_{meg}_exp-fwd.fif".format(dir=meg_dir,meg=meg))
    # load the csd matrices needed: i.e. for every freq band, get exp, ton, neg and pos (and maybe bas & rest)
    csd_exp_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_exp_alpha = csd_exp_alpha.mean()
    csd_exp_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_theta.h5".format(dir=meg_dir,meg=meg))
    csd_exp_theta = csd_exp_theta.mean()
    csd_exp_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_exp_beta_low = csd_exp_beta_low.mean()
    csd_exp_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_exp_beta_high = csd_exp_beta_high.mean()
    csd_exp_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_exp_gamma = csd_exp_gamma.mean()
    csd_ton_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_alpha = csd_ton_alpha.mean()
    csd_ton_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_theta = csd_ton_theta.mean()
    csd_ton_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_low = csd_ton_beta_low.mean()
    csd_ton_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_high = csd_ton_beta_high.mean()
    csd_ton_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma = csd_ton_gamma.mean()
    csd_neg_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_alpha = csd_neg_alpha.mean()
    csd_neg_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_theta = csd_neg_theta.mean()
    csd_neg_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_beta_low = csd_neg_beta_low.mean()
    csd_neg_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_beta_high = csd_neg_beta_high.mean()
    csd_neg_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_neg_gamma = csd_neg_gamma.mean()
    csd_pos_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_alpha = csd_pos_alpha.mean()
    csd_pos_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_theta = csd_pos_theta.mean()
    csd_pos_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_beta_low = csd_pos_beta_low.mean()
    csd_pos_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_beta_high = csd_pos_beta_high.mean()
    csd_pos_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_pos_gamma = csd_pos_gamma.mean()
    # csd_rest_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_alpha.h5".format(dir=meg_dir,meg=meg))
    # csd_rest_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_theta.h5".format(dir=meg_dir,meg=meg))
    # csd_rest_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_beta_low.h5".format(dir=meg_dir,meg=meg))
    # csd_rest_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_beta_high.h5".format(dir=meg_dir,meg=meg))
    # csd_rest_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_gamma.h5".format(dir=meg_dir,meg=meg))

    # make DICS beamformers / filters (common filters) for different freq bands
    filters_exp_alpha = make_dics(epo_exp.info,fwd_exp,csd_exp_alpha,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_theta = make_dics(epo_exp.info,fwd_exp,csd_exp_theta,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_beta_low = make_dics(epo_exp.info,fwd_exp,csd_exp_beta_low,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_beta_high = make_dics(epo_exp.info,fwd_exp,csd_exp_beta_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_gamma = make_dics(epo_exp.info,fwd_exp,csd_exp_gamma,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)

    # apply the DICS beamformers to get source Estimates for ton, neg, pos using common filters & save to file
    stc_ton_alpha, freqs_ton_alpha = apply_dics_csd(csd_ton_alpha,filters_exp_alpha)
    stc_ton_theta, freqs_ton_theta = apply_dics_csd(csd_ton_theta,filters_exp_theta)
    stc_ton_beta_low, freqs_ton_beta_low = apply_dics_csd(csd_ton_beta_low,filters_exp_beta_low)
    stc_ton_beta_high, freqs_ton_beta_high = apply_dics_csd(csd_ton_beta_high,filters_exp_beta_high)
    stc_ton_gamma, freqs_ton_gamma = apply_dics_csd(csd_ton_gamma,filters_exp_gamma)
    stc_ton_alpha.save(fname=meg_dir+"nc_{}_stc_ton_alpha".format(meg))
    stc_ton_theta.save(fname=meg_dir+"nc_{}_stc_ton_theta".format(meg))
    stc_ton_beta_low.save(fname=meg_dir+"nc_{}_stc_ton_beta_low".format(meg))
    stc_ton_beta_high.save(fname=meg_dir+"nc_{}_stc_ton_beta_high".format(meg))
    stc_ton_gamma.save(fname=meg_dir+"nc_{}_stc_ton_gamma".format(meg))

    stc_neg_alpha, freqs_neg_alpha = apply_dics_csd(csd_neg_alpha,filters_exp_alpha)
    stc_neg_theta, freqs_neg_theta = apply_dics_csd(csd_neg_theta,filters_exp_theta)
    stc_neg_beta_low, freqs_neg_beta_low = apply_dics_csd(csd_neg_beta_low,filters_exp_beta_low)
    stc_neg_beta_high, freqs_neg_beta_high = apply_dics_csd(csd_neg_beta_high,filters_exp_beta_high)
    stc_neg_gamma, freqs_neg_gamma = apply_dics_csd(csd_neg_gamma,filters_exp_gamma)
    stc_neg_alpha.save(fname=meg_dir+"nc_{}_stc_neg_alpha".format(meg))
    stc_neg_theta.save(fname=meg_dir+"nc_{}_stc_neg_theta".format(meg))
    stc_neg_beta_low.save(fname=meg_dir+"nc_{}_stc_neg_beta_low".format(meg))
    stc_neg_beta_high.save(fname=meg_dir+"nc_{}_stc_neg_beta_high".format(meg))
    stc_neg_gamma.save(fname=meg_dir+"nc_{}_stc_neg_gamma".format(meg))

    stc_pos_alpha, freqs_pos_alpha = apply_dics_csd(csd_pos_alpha,filters_exp_alpha)
    stc_pos_theta, freqs_pos_theta = apply_dics_csd(csd_pos_theta,filters_exp_theta)
    stc_pos_beta_low, freqs_pos_beta_low = apply_dics_csd(csd_pos_beta_low,filters_exp_beta_low)
    stc_pos_beta_high, freqs_pos_beta_high = apply_dics_csd(csd_pos_beta_high,filters_exp_beta_high)
    stc_pos_gamma, freqs_pos_gamma = apply_dics_csd(csd_pos_gamma,filters_exp_gamma)
    stc_pos_alpha.save(fname=meg_dir+"nc_{}_stc_pos_alpha".format(meg))
    stc_pos_theta.save(fname=meg_dir+"nc_{}_stc_pos_theta".format(meg))
    stc_pos_beta_low.save(fname=meg_dir+"nc_{}_stc_pos_beta_low".format(meg))
    stc_pos_beta_high.save(fname=meg_dir+"nc_{}_stc_pos_beta_high".format(meg))
    stc_pos_gamma.save(fname=meg_dir+"nc_{}_stc_pos_gamma".format(meg))

    # calculate the difference between conditions div. by baseline & save to file
    stc_diff_alpha = (stc_neg_alpha - stc_pos_alpha) / stc_ton_alpha
    stc_diff_theta = (stc_neg_theta - stc_pos_theta) / stc_ton_theta
    stc_diff_beta_low = (stc_neg_beta_low - stc_pos_beta_low) / stc_ton_beta_low
    stc_diff_beta_high = (stc_neg_beta_high - stc_pos_beta_high) / stc_ton_beta_high
    stc_diff_gamma = (stc_neg_gamma - stc_pos_gamma) / stc_ton_gamma
    stc_diff_alpha.save(fname=meg_dir+"nc_{}_stc_diff_alpha".format(meg))
    stc_diff_theta.save(fname=meg_dir+"nc_{}_stc_diff_theta".format(meg))
    stc_diff_beta_low.save(fname=meg_dir+"nc_{}_stc_diff_beta_low".format(meg))
    stc_diff_beta_high.save(fname=meg_dir+"nc_{}_stc_diff_beta_high".format(meg))
    stc_diff_gamma.save(fname=meg_dir+"nc_{}_stc_diff_gamma".format(meg))

    # morph the resulting stcs to fsaverage & save  (to be loaded again and averaged)
    #src = mne.read_source_spaces("{}nc_{}-src.fif".format(meg_dir,meg))  ## as no. of vertices doesn't match between src and stc (got reduced by building forward model with mindist 5.0) - stc is used directly for morphing
    morph = mne.compute_source_morph(stc_diff_alpha,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    stc_fsavg_diff_alpha = morph.apply(stc_diff_alpha)
    stc_fsavg_diff_theta = morph.apply(stc_diff_theta)
    stc_fsavg_diff_beta_low = morph.apply(stc_diff_beta_low)
    stc_fsavg_diff_beta_high = morph.apply(stc_diff_beta_high)
    stc_fsavg_diff_gamma = morph.apply(stc_diff_gamma)
    stc_fsavg_diff_alpha.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_alpha".format(meg))
    stc_fsavg_diff_theta.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_theta".format(meg))
    stc_fsavg_diff_beta_low.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_beta_low".format(meg))
    stc_fsavg_diff_beta_high.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_beta_high".format(meg))
    stc_fsavg_diff_gamma.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_gamma".format(meg))
