### merge this script with source prep II for new version with common filters ###

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
# sub_dict = {"NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
#            "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":list(np.arange(36,46))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":30}

for meg,mri in sub_dict.items():
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # separate the experimental conditions in the MEG data
    neg_a = epo_a["negative"]
    neg_b = epo_b["negative"]
    pos_a = epo_a["positive"]
    pos_b = epo_b["positive"]
    # calculate CSD matrix for gamma for each block
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    csd_rest_gamma = csd_morlet(rest, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_rest_gamma.save("{dir}nc_{meg}-csd_rest_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma = csd_morlet(ton, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_ton_gamma.save("{dir}nc_{meg}-csd_ton_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_a_gamma = csd_morlet(epo_a, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_a_gamma.save("{dir}nc_{meg}-csd_a_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_b_gamma = csd_morlet(epo_b, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_b_gamma.save("{dir}nc_{meg}-csd_b_gamma.h5".format(dir=meg_dir,meg=meg))
    # calculate CSDs for the experimental conditions in each block
    ## alternative with multitaper - template: csd_pos_a = csd_multitaper(pos_a,fmin=7,fmax=14,bandwidth=1)
    csd_pos_a_gamma = csd_morlet(pos_a, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_a_gamma.save("{dir}nc_{meg}-csd_pos_a_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_pos_b_gamma = csd_morlet(pos_b, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_pos_b_gamma.save("{dir}nc_{meg}-csd_pos_b_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_neg_a_gamma = csd_morlet(neg_a, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_a_gamma.save("{dir}nc_{meg}-csd_neg_a_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_neg_b_gamma = csd_morlet(neg_b, frequencies=freqs["gamma"], n_jobs=8, n_cycles=cycles["gamma"], decim=1)
    csd_neg_b_gamma.save("{dir}nc_{meg}-csd_neg_b_gamma.h5".format(dir=meg_dir,meg=meg))
    # calculate CSD matrix for theta for each block
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    csd_rest_theta = csd_morlet(rest, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_rest_theta.save("{dir}nc_{meg}-csd_rest_theta.h5".format(dir=meg_dir,meg=meg))
    csd_ton_theta = csd_morlet(ton, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_ton_theta.save("{dir}nc_{meg}-csd_ton_theta.h5".format(dir=meg_dir,meg=meg))
    csd_a_theta = csd_morlet(epo_a, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_a_theta.save("{dir}nc_{meg}-csd_a_theta.h5".format(dir=meg_dir,meg=meg))
    csd_b_theta = csd_morlet(epo_b, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_b_theta.save("{dir}nc_{meg}-csd_b_theta.h5".format(dir=meg_dir,meg=meg))
    # calculate CSDs for the experimental conditions in each block
    ## alternative with multitaper - template: csd_pos_a = csd_multitaper(pos_a,fmin=7,fmax=14,bandwidth=1)
    csd_pos_a_theta = csd_morlet(pos_a, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_a_theta.save("{dir}nc_{meg}-csd_pos_a_theta.h5".format(dir=meg_dir,meg=meg))
    csd_pos_b_theta = csd_morlet(pos_b, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_pos_b_theta.save("{dir}nc_{meg}-csd_pos_b_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_a_theta = csd_morlet(neg_a, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_a_theta.save("{dir}nc_{meg}-csd_neg_a_theta.h5".format(dir=meg_dir,meg=meg))
    csd_neg_b_theta = csd_morlet(neg_b, frequencies=freqs["theta"], n_jobs=8, n_cycles=cycles["theta"], decim=1)
    csd_neg_b_theta.save("{dir}nc_{meg}-csd_neg_b_theta.h5".format(dir=meg_dir,meg=meg))
    # calculate CSD matrix for low beta for each block
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    csd_rest_beta_low = csd_morlet(rest, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_rest_beta_low.save("{dir}nc_{meg}-csd_rest_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_low = csd_morlet(ton, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_ton_beta_low.save("{dir}nc_{meg}-csd_ton_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_a_beta_low = csd_morlet(epo_a, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_a_beta_low.save("{dir}nc_{meg}-csd_a_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_b_beta_low = csd_morlet(epo_b, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_b_beta_low.save("{dir}nc_{meg}-csd_b_beta_low.h5".format(dir=meg_dir,meg=meg))
    # calculate CSDs for the experimental conditions in each block
    ## alternative with multitaper - template: csd_pos_a = csd_multitaper(pos_a,fmin=7,fmax=14,bandwidth=1)
    csd_pos_a_beta_low = csd_morlet(pos_a, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_a_beta_low.save("{dir}nc_{meg}-csd_pos_a_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_pos_b_beta_low = csd_morlet(pos_b, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_pos_b_beta_low.save("{dir}nc_{meg}-csd_pos_b_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_a_beta_low = csd_morlet(neg_a, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_a_beta_low.save("{dir}nc_{meg}-csd_neg_a_beta_low.h5".format(dir=meg_dir,meg=meg))
    csd_neg_b_beta_low = csd_morlet(neg_b, frequencies=freqs["beta_low"], n_jobs=8, n_cycles=cycles["beta_low"], decim=1)
    csd_neg_b_beta_low.save("{dir}nc_{meg}-csd_neg_b_beta_low.h5".format(dir=meg_dir,meg=meg))
    # calculate CSD matrix for high beta for each block
    ## alternative with multitaper - template:  csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    csd_rest_beta_high = csd_morlet(rest, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_rest_beta_high.save("{dir}nc_{meg}-csd_rest_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_beta_high = csd_morlet(ton, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_ton_beta_high.save("{dir}nc_{meg}-csd_ton_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_a_beta_high = csd_morlet(epo_a, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_a_beta_high.save("{dir}nc_{meg}-csd_a_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_b_beta_high = csd_morlet(epo_b, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_b_beta_high.save("{dir}nc_{meg}-csd_b_beta_high.h5".format(dir=meg_dir,meg=meg))
    # calculate CSDs for the experimental conditions in each block
    ## alternative with multitaper - template: csd_pos_a = csd_multitaper(pos_a,fmin=7,fmax=14,bandwidth=1)
    csd_pos_a_beta_high = csd_morlet(pos_a, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_a_beta_high.save("{dir}nc_{meg}-csd_pos_a_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_b_beta_high = csd_morlet(pos_b, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_pos_b_beta_high.save("{dir}nc_{meg}-csd_pos_b_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_a_beta_high = csd_morlet(neg_a, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_a_beta_high.save("{dir}nc_{meg}-csd_neg_a_beta_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_b_beta_high = csd_morlet(neg_b, frequencies=freqs["beta_high"], n_jobs=8, n_cycles=cycles["beta_high"], decim=1)
    csd_neg_b_beta_high.save("{dir}nc_{meg}-csd_neg_b_beta_high.h5".format(dir=meg_dir,meg=meg))
