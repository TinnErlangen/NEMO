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
freqs = {"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}
cycles = {"gamma_high":35}

for meg,mri in sub_dict.items():
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = ton.info['dev_head_t']
    epo_bas = mne.concatenate_epochs([rest,ton])
    # load the forward models from each experimental block
    fwd_rest = mne.read_forward_solution("{dir}nc_{meg}_1-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_ton =  mne.read_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_base = mne.read_forward_solution("{dir}nc_{meg}_base-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_exp = mne.read_forward_solution("{dir}nc_{meg}_exp-fwd.fif".format(dir=meg_dir,meg=meg))
    # load the csd matrices needed
    csd_exp_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_exp_gamma_high = csd_exp_gamma_high.mean()
    csd_bas_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_bas_gamma_high = csd_bas_gamma_high.mean()
    csd_rest_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_rest_gamma_high = csd_rest_gamma_high.mean()
    csd_ton_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma_high = csd_ton_gamma_high.mean()
    csd_neg_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_gamma_high = csd_neg_gamma_high.mean()
    csd_pos_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_gamma_high = csd_pos_gamma_high.mean()
    # make DICS beamformers / filters (common filters)
    filters_exp_gamma_high = make_dics(epo_exp.info,fwd_exp,csd_exp_gamma_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    filters_bas_gamma_high = make_dics(epo_bas.info,fwd_base,csd_bas_gamma_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # apply the DICS beamformers to get source Estimates for ton, neg, pos using common filters & save to file
    # for experiment
    stc_ton_gamma_high, freqs_ton_gamma_high = apply_dics_csd(csd_ton_gamma_high,filters_exp_gamma_high)
    stc_ton_gamma_high.save(fname=meg_dir+"nc_{}_stc_ton_gamma_high".format(meg))
    stc_neg_gamma_high, freqs_neg_gamma_high = apply_dics_csd(csd_neg_gamma_high,filters_exp_gamma_high)
    stc_neg_gamma_high.save(fname=meg_dir+"nc_{}_stc_neg_gamma_high".format(meg))
    stc_pos_gamma_high, freqs_pos_gamma_high = apply_dics_csd(csd_pos_gamma_high,filters_exp_gamma_high)
    stc_pos_gamma_high.save(fname=meg_dir+"nc_{}_stc_pos_gamma_high".format(meg))
    # calculate the difference between conditions div. by baseline & save to file
    stc_diff_gamma_high = (stc_neg_gamma_high - stc_pos_gamma_high) / stc_ton_gamma_high
    stc_diff_gamma_high.save(fname=meg_dir+"nc_{}_stc_diff_gamma_high".format(meg))
    # apply the DICS beamformers to get source Estimates for ton, rest using common filters & save to file
    # for tone baseline
    stc_tonb_gamma_high, freqs_tonb_gamma_high = apply_dics_csd(csd_ton_gamma_high,filters_bas_gamma_high)
    stc_tonb_gamma_high.save(fname=meg_dir+"nc_{}_stc_tonb_gamma_high".format(meg))
    stc_rest_gamma_high, freqs_rest_gamma_high = apply_dics_csd(csd_rest_gamma_high,filters_bas_gamma_high)
    stc_rest_gamma_high.save(fname=meg_dir+"nc_{}_stc_restbas_gamma_high".format(meg))
    # calculate the difference between conditions div. by baseline & save to file
    stc_tonbas_gamma_high = (stc_tonb_gamma_high - stc_rest_gamma_high) / stc_rest_gamma_high
    stc_tonbas_gamma_high.save(fname=meg_dir+"nc_{}_stc_tonbas_gamma_high".format(meg))

    # morph the resulting stcs to fsaverage & save
    morph = mne.compute_source_morph(stc_diff_gamma_high,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    stc_fsavg_diff_gamma_high = morph.apply(stc_diff_gamma_high)
    stc_fsavg_diff_gamma_high.save(fname=meg_dir+"nc_{}_stc_fsavg_diff_gamma_high".format(meg))
    morph = mne.compute_source_morph(stc_tonbas_gamma_high,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    stc_fsavg_tonbas_gamma_high = morph.apply(stc_tonbas_gamma_high)
    stc_fsavg_tonbas_gamma_high.save(fname=meg_dir+"nc_{}_stc_fsavg_tonbas_gamma_high".format(meg))
