# calculate and save CSDs for all conditions and combined data * new high gamma freq band

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
import random

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
#sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",}  # these are done
sub_dict = {"NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48",
           "NEM_21":"WKI71_fa","NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72","NEM_30":"DIU11","NEM_31":"BLE94",
           "NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa",
           "NEM_37":"EAM67"}
#sub_dict = {"NEM_10":"GIZ04"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56))}
freqs = {"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}
cycles = {"gamma_high":35}

for meg,mri in sub_dict.items():
    # load and prepare the MEG data for rest and ton
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = ton.info['dev_head_t']
    epo_bas = mne.concatenate_epochs([rest,ton])
    epo_exp = mne.read_epochs("{dir}nc_{sub}_exp-epo.fif".format(dir=meg_dir,sub=meg))
    #separate positive and negative condition epochs
    pos = epo_exp['positive']
    neg = epo_exp['negative']
    # calculate & save CSDs
    csd_bas_gamma_high = csd_morlet(epo_bas, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_bas_gamma_high.save("{dir}nc_{meg}-csd_bas_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_exp_gamma_high = csd_morlet(epo_exp, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_exp_gamma_high.save("{dir}nc_{meg}-csd_exp_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_rest_gamma_high = csd_morlet(rest, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_rest_gamma_high.save("{dir}nc_{meg}-csd_rest_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma_high = csd_morlet(ton, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_ton_gamma_high.save("{dir}nc_{meg}-csd_ton_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_gamma_high = csd_morlet(neg, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_neg_gamma_high.save("{dir}nc_{meg}-csd_neg_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_gamma_high = csd_morlet(pos, frequencies=freqs["gamma_high"], n_jobs=8, n_cycles=cycles["gamma_high"], decim=1)
    csd_pos_gamma_high.save("{dir}nc_{meg}-csd_pos_gamma_high.h5".format(dir=meg_dir,meg=meg))
