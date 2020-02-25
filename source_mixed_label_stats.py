## preparation of source space and forward solutions as mixed source space with cortex and subcortical volume structures

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_17":"DEN59","NEM_18":"SAG13",
           "NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72",
           "NEM_34":"KER27","NEM_36":"BRA52_fa","NEM_16":"KIO12","NEM_20":"PAG48","NEM_31":"BLE94","NEM_35":"MUN79"}
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23",}
# freqs = {"theta":list(np.arange(4,6)),"alpha":list(np.arange(8,13)),"beta_low":list(np.arange(17,23)),
         # "beta_high":list(np.arange(26,34))}

# setup array for group ANALYSES
# for experiment
X_label_alpha_diff = []
X_label_theta_diff = []
X_label_beta_low_diff = []
X_label_beta_high_diff = []
X_label_gamma_diff = []
# for baseline
X_label_alpha_tonbas = []
X_label_theta_tonbas = []
X_label_beta_low_tonbas = []
X_label_beta_high_tonbas = []
X_label_gamma_tonbas = []

for meg,mri in sub_dict.items():
    # load mixed fwds
    fwd_base = mne.read_forward_solution("{dir}nc_{meg}_limb_mix_base-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_exp = mne.read_forward_solution("{dir}nc_{meg}_limb_mix_exp-fwd.fif".format(dir=meg_dir,meg=meg))
    # load needed epos
    epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = ton.info['dev_head_t']
    epo_bas = mne.concatenate_epochs([rest,ton])
    del rest, ton
    # load needed CSDs
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
    csd_bas_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_bas_alpha = csd_bas_alpha.mean()
    csd_bas_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_theta.h5".format(dir=meg_dir,meg=meg))
    csd_bas_theta = csd_bas_theta.mean()
    csd_bas_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_bas_beta_low = csd_bas_beta_low.mean()
    csd_bas_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_bas_beta_high = csd_bas_beta_high.mean()
    csd_bas_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_bas_gamma = csd_bas_gamma.mean()
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
    csd_rest_alpha = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_rest_alpha = csd_rest_alpha.mean()
    csd_rest_theta = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_theta.h5".format(dir=meg_dir,meg=meg))
    csd_rest_theta = csd_rest_theta.mean()
    csd_rest_beta_low = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_rest_beta_low = csd_rest_beta_low.mean()
    csd_rest_beta_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_rest_beta_high = csd_rest_beta_high.mean()
    csd_rest_gamma = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_rest_gamma = csd_rest_gamma.mean()
    # build DICS filters
    # for experimental conditions analyses
    filters_exp_alpha = make_dics(epo_exp.info,fwd_exp,csd_exp_alpha,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_theta = make_dics(epo_exp.info,fwd_exp,csd_exp_theta,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_beta_low = make_dics(epo_exp.info,fwd_exp,csd_exp_beta_low,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_beta_high = make_dics(epo_exp.info,fwd_exp,csd_exp_beta_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_exp_gamma = make_dics(epo_exp.info,fwd_exp,csd_exp_gamma,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # clean memory
    del epo_exp, fwd_exp, csd_exp_alpha, csd_exp_theta, csd_exp_beta_low, csd_exp_beta_high, csd_exp_gamma
    # for baseline analyses
    filters_bas_alpha = make_dics(epo_bas.info,fwd_base,csd_bas_alpha,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas_theta = make_dics(epo_bas.info,fwd_base,csd_bas_theta,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas_beta_low = make_dics(epo_bas.info,fwd_base,csd_bas_beta_low,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas_beta_high = make_dics(epo_bas.info,fwd_base,csd_bas_beta_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas_gamma = make_dics(epo_bas.info,fwd_base,csd_bas_gamma,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # clean memory
    del epo_bas, fwd_base, csd_bas_alpha, csd_bas_theta, csd_bas_beta_low, csd_bas_beta_high, csd_bas_gamma
    # apply filters, calculate difference stcs & save
    # for experiment
    stc_ton_alpha, freqs_ton_alpha = apply_dics_csd(csd_ton_alpha,filters_exp_alpha)
    stc_ton_theta, freqs_ton_theta = apply_dics_csd(csd_ton_theta,filters_exp_theta)
    stc_ton_beta_low, freqs_ton_beta_low = apply_dics_csd(csd_ton_beta_low,filters_exp_beta_low)
    stc_ton_beta_high, freqs_ton_beta_high = apply_dics_csd(csd_ton_beta_high,filters_exp_beta_high)
    stc_ton_gamma, freqs_ton_gamma = apply_dics_csd(csd_ton_gamma,filters_exp_gamma)
    stc_neg_alpha, freqs_neg_alpha = apply_dics_csd(csd_neg_alpha,filters_exp_alpha)
    stc_neg_theta, freqs_neg_theta = apply_dics_csd(csd_neg_theta,filters_exp_theta)
    stc_neg_beta_low, freqs_neg_beta_low = apply_dics_csd(csd_neg_beta_low,filters_exp_beta_low)
    stc_neg_beta_high, freqs_neg_beta_high = apply_dics_csd(csd_neg_beta_high,filters_exp_beta_high)
    stc_neg_gamma, freqs_neg_gamma = apply_dics_csd(csd_neg_gamma,filters_exp_gamma)
    stc_pos_alpha, freqs_pos_alpha = apply_dics_csd(csd_pos_alpha,filters_exp_alpha)
    stc_pos_theta, freqs_pos_theta = apply_dics_csd(csd_pos_theta,filters_exp_theta)
    stc_pos_beta_low, freqs_pos_beta_low = apply_dics_csd(csd_pos_beta_low,filters_exp_beta_low)
    stc_pos_beta_high, freqs_pos_beta_high = apply_dics_csd(csd_pos_beta_high,filters_exp_beta_high)
    stc_pos_gamma, freqs_pos_gamma = apply_dics_csd(csd_pos_gamma,filters_exp_gamma)
    stc_diff_alpha = (stc_neg_alpha - stc_pos_alpha) / stc_ton_alpha
    stc_diff_theta = (stc_neg_theta - stc_pos_theta) / stc_ton_theta
    stc_diff_beta_low = (stc_neg_beta_low - stc_pos_beta_low) / stc_ton_beta_low
    stc_diff_beta_high = (stc_neg_beta_high - stc_pos_beta_high) / stc_ton_beta_high
    stc_diff_gamma = (stc_neg_gamma - stc_pos_gamma) / stc_ton_gamma
    stc_diff_alpha.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_alpha".format(meg))
    stc_diff_theta.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_theta".format(meg))
    stc_diff_beta_low.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_beta_low".format(meg))
    stc_diff_beta_high.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_beta_high".format(meg))
    stc_diff_gamma.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_gamma".format(meg))
    # clean memory
    del filters_exp_alpha, filters_exp_theta, filters_exp_beta_low, filters_exp_beta_high, filters_exp_gamma
    del csd_neg_alpha, csd_neg_theta, csd_neg_beta_low, csd_neg_beta_high, csd_neg_gamma
    del csd_pos_alpha, csd_pos_theta, csd_pos_beta_low, csd_pos_beta_high, csd_pos_gamma
    # for baseline
    stc_ton_alpha, freqs_ton_alpha = apply_dics_csd(csd_ton_alpha,filters_bas_alpha)
    stc_ton_theta, freqs_ton_theta = apply_dics_csd(csd_ton_theta,filters_bas_theta)
    stc_ton_beta_low, freqs_ton_beta_low = apply_dics_csd(csd_ton_beta_low,filters_bas_beta_low)
    stc_ton_beta_high, freqs_ton_beta_high = apply_dics_csd(csd_ton_beta_high,filters_bas_beta_high)
    stc_ton_gamma, freqs_ton_gamma = apply_dics_csd(csd_ton_gamma,filters_bas_gamma)
    stc_rest_alpha, freqs_rest_alpha = apply_dics_csd(csd_rest_alpha,filters_bas_alpha)
    stc_rest_theta, freqs_rest_theta = apply_dics_csd(csd_rest_theta,filters_bas_theta)
    stc_rest_beta_low, freqs_rest_beta_low = apply_dics_csd(csd_rest_beta_low,filters_bas_beta_low)
    stc_rest_beta_high, freqs_rest_beta_high = apply_dics_csd(csd_rest_beta_high,filters_bas_beta_high)
    stc_rest_gamma, freqs_rest_gamma = apply_dics_csd(csd_rest_gamma,filters_bas_gamma)
    stc_tonbas_alpha = (stc_ton_alpha - stc_rest_alpha) / stc_rest_alpha
    stc_tonbas_theta = (stc_ton_theta - stc_rest_theta) / stc_rest_theta
    stc_tonbas_beta_low = (stc_ton_beta_low - stc_rest_beta_low) / stc_rest_beta_low
    stc_tonbas_beta_high = (stc_ton_beta_high - stc_rest_beta_high) / stc_rest_beta_high
    stc_tonbas_gamma = (stc_ton_gamma - stc_rest_gamma) / stc_rest_gamma
    stc_tonbas_alpha.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_alpha".format(meg))
    stc_tonbas_theta.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_theta".format(meg))
    stc_tonbas_beta_low.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_beta_low".format(meg))
    stc_tonbas_beta_high.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_beta_high".format(meg))
    stc_tonbas_gamma.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_gamma".format(meg))
    # clean memory
    del filters_bas_alpha, filters_bas_theta, filters_bas_beta_low, filters_bas_beta_high, filters_bas_gamma
    del csd_rest_alpha, csd_rest_theta, csd_rest_beta_low, csd_rest_beta_high, csd_rest_gamma
    del csd_ton_alpha, csd_ton_theta, csd_ton_beta_low, csd_ton_beta_high, csd_ton_gamma
    del stc_rest_alpha, stc_rest_theta, stc_rest_beta_low, stc_rest_beta_high, stc_rest_gamma
    del stc_ton_alpha, stc_ton_theta, stc_ton_beta_low, stc_ton_beta_high, stc_ton_gamma
    del stc_neg_alpha, stc_neg_theta, stc_neg_beta_low, stc_neg_beta_high, stc_neg_gamma
    del stc_pos_alpha, stc_pos_theta, stc_pos_beta_low, stc_pos_beta_high, stc_pos_gamma
    # read mixed source space
    src = mne.read_source_spaces("{}nc_{}_limb_mix-src.fif".format(meg_dir,meg))
    # load labels
    labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
                   'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']
    # labels_parc = mne.read_labels_from_annot(mri, parc='aparc', subjects_dir=mri_dir)
    labels_dest = mne.read_labels_from_annot(mri, parc='aparc.a2009s', subjects_dir=mri_dir)
    # for each difference STC , extract a label_tc, get values out and sorted into group array
    # for experiment
    label_tc_alpha = mne.extract_label_time_course([stc_diff_alpha],labels_dest,src,mode='mean')
    X_label_alpha_diff.append(label_tc_alpha[0])
    del label_tc_alpha
    label_tc_theta = mne.extract_label_time_course([stc_diff_theta],labels_dest,src,mode='mean')
    X_label_theta_diff.append(label_tc_theta[0])
    del label_tc_theta
    label_tc_beta_low = mne.extract_label_time_course([stc_diff_beta_low],labels_dest,src,mode='mean')
    X_label_beta_low_diff.append(label_tc_beta_low[0])
    del label_tc_beta_low
    label_tc_beta_high = mne.extract_label_time_course([stc_diff_beta_high],labels_dest,src,mode='mean')
    X_label_beta_high_diff.append(label_tc_beta_high[0])
    del label_tc_beta_high
    label_tc_gamma = mne.extract_label_time_course([stc_diff_gamma],labels_dest,src,mode='mean')
    X_label_gamma_diff.append(label_tc_gamma[0])
    del label_tc_gamma
    # for baseline
    label_tc_alpha_tonbas = mne.extract_label_time_course([stc_tonbas_alpha],labels_dest,src,mode='mean')
    X_label_alpha_tonbas.append(label_tc_alpha_tonbas[0])
    del label_tc_alpha_tonbas
    label_tc_theta_tonbas = mne.extract_label_time_course([stc_tonbas_theta],labels_dest,src,mode='mean')
    X_label_theta_tonbas.append(label_tc_theta_tonbas[0])
    del label_tc_theta_tonbas
    label_tc_beta_low_tonbas = mne.extract_label_time_course([stc_tonbas_beta_low],labels_dest,src,mode='mean')
    X_label_beta_low_tonbas.append(label_tc_beta_low_tonbas[0])
    del label_tc_beta_low_tonbas
    label_tc_beta_high_tonbas = mne.extract_label_time_course([stc_tonbas_beta_high],labels_dest,src,mode='mean')
    X_label_beta_high_tonbas.append(label_tc_beta_high_tonbas[0])
    del label_tc_beta_high_tonbas
    label_tc_gamma_tonbas = mne.extract_label_time_course([stc_tonbas_gamma],labels_dest,src,mode='mean')
    X_label_gamma_tonbas.append(label_tc_gamma_tonbas[0])
    del label_tc_gamma_tonbas

# group permutation analyses
## how do I get connectivity here ???? between the labels ... either I produce a matrix myself ... or use permutation_cluster_test with connectivity=False (see API) --> doing this first now
# prepare source alpha diff permutation t-test
# src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
# connectivity = mne.spatial_src_connectivity(src)

# alpha for exp_diff and tonbas
X_label_alpha_diff = np.array(X_label_alpha_diff)
a_t_obs, a_pvals, a_H0 = mne.stats.permutation_t_test(X_label_alpha_diff, n_permutations=10000, tail=0, n_jobs=4, seed=None)
a_good_pval_inds = np.where(a_pvals < 0.05)[0]

X_label_alpha_tonbas = np.array(X_label_alpha_tonbas)
a_ton_t_obs, a_ton_pvals, a_ton_H0 = mne.stats.permutation_t_test(X_label_alpha_tonbas, n_permutations=10000, tail=0, n_jobs=4, seed=None)
a_ton_good_pval_inds = np.where(a_ton_pvals < 0.05)[0]
