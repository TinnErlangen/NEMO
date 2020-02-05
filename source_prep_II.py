# calculate and save CSDs for all conditions and combined data * all frequency bands of interest

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

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
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = ton.info['dev_head_t']
    epo_a.info['dev_head_t'] = ton.info['dev_head_t']
    epo_b.info['dev_head_t'] = ton.info['dev_head_t']
    epo_all = mne.concatenate_epochs([rest,ton,epo_a,epo_b])
    epo_exp = mne.concatenate_epochs([ton,epo_a,epo_b])
    epo_bas = mne.concatenate_epochs([rest,ton])
    epo_r1 = mne.concatenate_epochs([epo_all['ton_r1'],epo_all['r1']])
    epo_r2 = mne.concatenate_epochs([epo_all['ton_r2'],epo_all['r2']])
    epo_s1 = mne.concatenate_epochs([epo_all['ton_s1'],epo_all['s1']])
    epo_s2 = mne.concatenate_epochs([epo_all['ton_s2'],epo_all['s2']])

    # calculate CSD matrix for each all, base, exp, rest
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)

    csd_all_alpha = csd_morlet(epo_all, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_all_alpha.save("{dir}nc_{meg}-csd_all_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_all_theta = csd_morlet(epo_all, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_all_theta.save("{dir}nc_{meg}-csd_all_theta.h5".format(dir=meg_dir,meg=meg))
    csd_all_beta_low = csd_morlet(epo_all, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_all_beta_low.save("{dir}nc_{meg}-csd_all_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_all_beta_high = csd_morlet(epo_all, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_all_beta_high.save("{dir}nc_{meg}-csd_all_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_all_gamma = csd_morlet(epo_all, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_all_gamma.save("{dir}nc_{meg}-csd_all_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_bas_alpha = csd_morlet(epo_bas, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_bas_alpha.save("{dir}nc_{meg}-csd_bas_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_bas_theta = csd_morlet(epo_bas, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_bas_theta.save("{dir}nc_{meg}-csd_bas_theta.h5".format(dir=meg_dir,meg=meg))
    csd_bas_beta_low = csd_morlet(epo_bas, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_bas_beta_low.save("{dir}nc_{meg}-csd_bas_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_bas_beta_high = csd_morlet(epo_bas, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_bas_beta_high.save("{dir}nc_{meg}-csd_bas_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_bas_gamma = csd_morlet(epo_bas, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_bas_gamma.save("{dir}nc_{meg}-csd_bas_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_exp_alpha = csd_morlet(epo_exp, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_exp_alpha.save("{dir}nc_{meg}-csd_exp_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_exp_theta = csd_morlet(epo_exp, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_exp_theta.save("{dir}nc_{meg}-csd_exp_theta.h5".format(dir=meg_dir,meg=meg))
    csd_exp_beta_low = csd_morlet(epo_exp, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_exp_beta_low.save("{dir}nc_{meg}-csd_exp_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_exp_beta_high = csd_morlet(epo_exp, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_exp_beta_high.save("{dir}nc_{meg}-csd_exp_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_exp_gamma = csd_morlet(epo_exp, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_exp_gamma.save("{dir}nc_{meg}-csd_exp_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_rest_alpha = csd_morlet(rest, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_rest_alpha.save("{dir}nc_{meg}-csd_rest_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_rest_theta = csd_morlet(rest, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_rest_theta.save("{dir}nc_{meg}-csd_rest_theta.h5".format(dir=meg_dir,meg=meg))
    csd_rest_beta_low = csd_morlet(rest, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_rest_beta_low.save("{dir}nc_{meg}-csd_rest_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_rest_beta_high = csd_morlet(rest, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_rest_beta_high.save("{dir}nc_{meg}-csd_rest_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_rest_gamma = csd_morlet(rest, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_rest_gamma.save("{dir}nc_{meg}-csd_rest_gamma.h5".format(dir=meg_dir,meg=meg))

    # calculate CSD matrix for each ton, neg, pos, as well as sounds r1,r2,s1,s2
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)

    csd_ton_alpha = csd_morlet(ton, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_ton_alpha.save("{dir}nc_{meg}-csd_ton_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_theta = csd_morlet(ton, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_theta.save("{dir}nc_{meg}-csd_ton_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_low = csd_morlet(ton, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_beta_low.save("{dir}nc_{meg}-csd_ton_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_high = csd_morlet(ton, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_beta_high.save("{dir}nc_{meg}-csd_ton_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma = csd_morlet(ton, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_gamma.save("{dir}nc_{meg}-csd_ton_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_neg_alpha = csd_morlet(epo_all['negative'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_neg_alpha.save("{dir}nc_{meg}-csd_neg_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_theta = csd_morlet(epo_all['negative'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_theta.save("{dir}nc_{meg}-csd_neg_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_beta_low = csd_morlet(epo_all['negative'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_beta_low.save("{dir}nc_{meg}-csd_neg_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_beta_high = csd_morlet(epo_all['negative'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_beta_high.save("{dir}nc_{meg}-csd_neg_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_gamma = csd_morlet(epo_all['negative'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_gamma.save("{dir}nc_{meg}-csd_neg_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_pos_alpha = csd_morlet(epo_all['positive'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_pos_alpha.save("{dir}nc_{meg}-csd_pos_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_theta = csd_morlet(epo_all['positive'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_theta.save("{dir}nc_{meg}-csd_pos_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_beta_low = csd_morlet(epo_all['positive'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_beta_low.save("{dir}nc_{meg}-csd_pos_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_beta_high = csd_morlet(epo_all['positive'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_beta_high.save("{dir}nc_{meg}-csd_pos_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_gamma = csd_morlet(epo_all['positive'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_gamma.save("{dir}nc_{meg}-csd_pos_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_r1_alpha = csd_morlet(epo_r1, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_r1_alpha.save("{dir}nc_{meg}-csd_r1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_r1_theta = csd_morlet(epo_r1, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_r1_theta.save("{dir}nc_{meg}-csd_r1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_r1_beta_low = csd_morlet(epo_r1, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_r1_beta_low.save("{dir}nc_{meg}-csd_r1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_r1_beta_high = csd_morlet(epo_r1, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_r1_beta_high.save("{dir}nc_{meg}-csd_r1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_r1_gamma = csd_morlet(epo_r1, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_r1_gamma.save("{dir}nc_{meg}-csd_r1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_r2_alpha = csd_morlet(epo_r2, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_r2_alpha.save("{dir}nc_{meg}-csd_r2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_r2_theta = csd_morlet(epo_r2, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_r2_theta.save("{dir}nc_{meg}-csd_r2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_r2_beta_low = csd_morlet(epo_r2, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_r2_beta_low.save("{dir}nc_{meg}-csd_r2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_r2_beta_high = csd_morlet(epo_r2, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_r2_beta_high.save("{dir}nc_{meg}-csd_r2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_r2_gamma = csd_morlet(epo_r2, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_r2_gamma.save("{dir}nc_{meg}-csd_r2_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_s1_alpha = csd_morlet(epo_s1, frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_s1_alpha.save("{dir}nc_{meg}-csd_s1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_s1_theta = csd_morlet(epo_s1, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_s1_theta.save("{dir}nc_{meg}-csd_s1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_s1_beta_low = csd_morlet(epo_s1, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_s1_beta_low.save("{dir}nc_{meg}-csd_s1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_s1_beta_high = csd_morlet(epo_s1, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_s1_beta_high.save("{dir}nc_{meg}-csd_s1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_s1_gamma = csd_morlet(epo_s1, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_s1_gamma.save("{dir}nc_{meg}-csd_s1_gamma.h5".format(dir=meg_dir,meg=meg))

    # calculate CSD matrix for each ton, neg, pos X r1,r2,s1,s2 condition
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    # ton subconds

    csd_ton_r1_alpha = csd_morlet(ton['ton_r1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_ton_r1_alpha.save("{dir}nc_{meg}-csd_ton_r1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r1_theta = csd_morlet(ton['ton_r1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_r1_theta.save("{dir}nc_{meg}-csd_ton_r1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r1_beta_low = csd_morlet(ton['ton_r1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_r1_beta_low.save("{dir}nc_{meg}-csd_ton_r1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r1_beta_high = csd_morlet(ton['ton_r1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_r1_beta_high.save("{dir}nc_{meg}-csd_ton_r1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r1_gamma = csd_morlet(ton['ton_r1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_r1_gamma.save("{dir}nc_{meg}-csd_ton_r1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_ton_r2_alpha = csd_morlet(ton['ton_r2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_ton_r2_alpha.save("{dir}nc_{meg}-csd_ton_r2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r2_theta = csd_morlet(ton['ton_r2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_r2_theta.save("{dir}nc_{meg}-csd_ton_r2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r2_beta_low = csd_morlet(ton['ton_r2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_r2_beta_low.save("{dir}nc_{meg}-csd_ton_r2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r2_beta_high = csd_morlet(ton['ton_r2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_r2_beta_high.save("{dir}nc_{meg}-csd_ton_r2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_r2_gamma = csd_morlet(ton['ton_r2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_r2_gamma.save("{dir}nc_{meg}-csd_ton_r2_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_ton_s1_alpha = csd_morlet(ton['ton_s1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_ton_s1_alpha.save("{dir}nc_{meg}-csd_ton_s1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s1_theta = csd_morlet(ton['ton_s1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_s1_theta.save("{dir}nc_{meg}-csd_ton_s1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s1_beta_low = csd_morlet(ton['ton_s1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_s1_beta_low.save("{dir}nc_{meg}-csd_ton_s1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s1_beta_high = csd_morlet(ton['ton_s1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_s1_beta_high.save("{dir}nc_{meg}-csd_ton_s1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s1_gamma = csd_morlet(ton['ton_s1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_s1_gamma.save("{dir}nc_{meg}-csd_ton_s1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_ton_s2_alpha = csd_morlet(ton['ton_s2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_ton_s2_alpha.save("{dir}nc_{meg}-csd_ton_s2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s2_theta = csd_morlet(ton['ton_s2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_s2_theta.save("{dir}nc_{meg}-csd_ton_s2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s2_beta_low = csd_morlet(ton['ton_s2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_s2_beta_low.save("{dir}nc_{meg}-csd_ton_s2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s2_beta_high = csd_morlet(ton['ton_s2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_s2_beta_high.save("{dir}nc_{meg}-csd_ton_s2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_s2_gamma = csd_morlet(ton['ton_s2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_s2_gamma.save("{dir}nc_{meg}-csd_ton_s2_gamma.h5".format(dir=meg_dir,meg=meg))

    # neg subconds
    csd_neg_r1_alpha = csd_morlet(epo_all['negative/r1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_neg_r1_alpha.save("{dir}nc_{meg}-csd_neg_r1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r1_theta = csd_morlet(epo_all['negative/r1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_r1_theta.save("{dir}nc_{meg}-csd_neg_r1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r1_beta_low = csd_morlet(epo_all['negative/r1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_r1_beta_low.save("{dir}nc_{meg}-csd_neg_r1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r1_beta_high = csd_morlet(epo_all['negative/r1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_r1_beta_high.save("{dir}nc_{meg}-csd_neg_r1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r1_gamma = csd_morlet(epo_all['negative/r1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_r1_gamma.save("{dir}nc_{meg}-csd_neg_r1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_neg_r2_alpha = csd_morlet(epo_all['negative/r2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_neg_r2_alpha.save("{dir}nc_{meg}-csd_neg_r2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r2_theta = csd_morlet(epo_all['negative/r2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_r2_theta.save("{dir}nc_{meg}-csd_neg_r2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r2_beta_low = csd_morlet(epo_all['negative/r2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_r2_beta_low.save("{dir}nc_{meg}-csd_neg_r2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r2_beta_high = csd_morlet(epo_all['negative/r2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_r2_beta_high.save("{dir}nc_{meg}-csd_neg_r2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_r2_gamma = csd_morlet(epo_all['negative/r2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_r2_gamma.save("{dir}nc_{meg}-csd_neg_r2_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_neg_s1_alpha = csd_morlet(epo_all['negative/s1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_neg_s1_alpha.save("{dir}nc_{meg}-csd_neg_s1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s1_theta = csd_morlet(epo_all['negative/s1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_s1_theta.save("{dir}nc_{meg}-csd_neg_s1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s1_beta_low = csd_morlet(epo_all['negative/s1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_s1_beta_low.save("{dir}nc_{meg}-csd_neg_s1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s1_beta_high = csd_morlet(epo_all['negative/s1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_s1_beta_high.save("{dir}nc_{meg}-csd_neg_s1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s1_gamma = csd_morlet(epo_all['negative/s1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_s1_gamma.save("{dir}nc_{meg}-csd_neg_s1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_neg_s2_alpha = csd_morlet(epo_all['negative/s2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_neg_s2_alpha.save("{dir}nc_{meg}-csd_neg_s2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s2_theta = csd_morlet(epo_all['negative/s2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_s2_theta.save("{dir}nc_{meg}-csd_neg_s2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s2_beta_low = csd_morlet(epo_all['negative/s2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_s2_beta_low.save("{dir}nc_{meg}-csd_neg_s2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s2_beta_high = csd_morlet(epo_all['negative/s2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_s2_beta_high.save("{dir}nc_{meg}-csd_neg_s2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_s2_gamma = csd_morlet(epo_all['negative/s2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_s2_gamma.save("{dir}nc_{meg}-csd_neg_s2_gamma.h5".format(dir=meg_dir,meg=meg))

    # pos subconds
    csd_pos_r1_alpha = csd_morlet(epo_all['positive/r1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_pos_r1_alpha.save("{dir}nc_{meg}-csd_pos_r1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r1_theta = csd_morlet(epo_all['positive/r1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_r1_theta.save("{dir}nc_{meg}-csd_pos_r1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r1_beta_low = csd_morlet(epo_all['positive/r1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_r1_beta_low.save("{dir}nc_{meg}-csd_pos_r1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r1_beta_high = csd_morlet(epo_all['positive/r1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_r1_beta_high.save("{dir}nc_{meg}-csd_pos_r1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r1_gamma = csd_morlet(epo_all['positive/r1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_r1_gamma.save("{dir}nc_{meg}-csd_pos_r1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_pos_r2_alpha = csd_morlet(epo_all['positive/r2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_pos_r2_alpha.save("{dir}nc_{meg}-csd_pos_r2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r2_theta = csd_morlet(epo_all['positive/r2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_r2_theta.save("{dir}nc_{meg}-csd_pos_r2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r2_beta_low = csd_morlet(epo_all['positive/r2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_r2_beta_low.save("{dir}nc_{meg}-csd_pos_r2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r2_beta_high = csd_morlet(epo_all['positive/r2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_r2_beta_high.save("{dir}nc_{meg}-csd_pos_r2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_r2_gamma = csd_morlet(epo_all['positive/r2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_r2_gamma.save("{dir}nc_{meg}-csd_pos_r2_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_pos_s1_alpha = csd_morlet(epo_all['positive/s1'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_pos_s1_alpha.save("{dir}nc_{meg}-csd_pos_s1_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s1_theta = csd_morlet(epo_all['positive/s1'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_s1_theta.save("{dir}nc_{meg}-csd_pos_s1_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s1_beta_low = csd_morlet(epo_all['positive/s1'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_s1_beta_low.save("{dir}nc_{meg}-csd_pos_s1_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s1_beta_high = csd_morlet(epo_all['positive/s1'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_s1_beta_high.save("{dir}nc_{meg}-csd_pos_s1_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s1_gamma = csd_morlet(epo_all['positive/s1'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_s1_gamma.save("{dir}nc_{meg}-csd_pos_s1_gamma.h5".format(dir=meg_dir,meg=meg))

    csd_pos_s2_alpha = csd_morlet(epo_all['positive/s2'], frequencies=freqs["alpha"], n_jobs=8, n_cycles=cycles["alpha"], decim=1)
    csd_pos_s2_alpha.save("{dir}nc_{meg}-csd_pos_s2_alpha.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s2_theta = csd_morlet(epo_all['positive/s2'], frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_s2_theta.save("{dir}nc_{meg}-csd_pos_s2_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s2_beta_low = csd_morlet(epo_all['positive/s2'], frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_s2_beta_low.save("{dir}nc_{meg}-csd_pos_s2_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s2_beta_high = csd_morlet(epo_all['positive/s2'], frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_s2_beta_high.save("{dir}nc_{meg}-csd_pos_s2_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_s2_gamma = csd_morlet(epo_all['positive/s2'], frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_s2_gamma.save("{dir}nc_{meg}-csd_pos_s2_gamma.h5".format(dir=meg_dir,meg=meg))
