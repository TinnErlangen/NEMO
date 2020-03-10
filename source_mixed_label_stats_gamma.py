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

# setup array for group ANALYSES
# for experiment
X_label_gamma_high_diff = []
# for baseline
X_label_gamma_high_tonbas = []

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
    csd_exp_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_exp_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_exp_gamma_high = csd_exp_gamma_high.mean()
    csd_bas_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_bas_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_bas_gamma_high = csd_bas_gamma_high.mean()
    csd_ton_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_ton_gamma_high = csd_ton_gamma_high.mean()
    csd_neg_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_neg_gamma_high = csd_neg_gamma_high.mean()
    csd_pos_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_pos_gamma_high = csd_pos_gamma_high.mean()
    csd_rest_gamma_high = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_gamma_high.h5".format(dir=meg_dir,meg=meg))
    csd_rest_gamma_high = csd_rest_gamma_high.mean()
    # build DICS filters
    # for experimental conditions analyses
    filters_exp_gamma_high = make_dics(epo_exp.info,fwd_exp,csd_exp_gamma_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # clean memory
    del epo_exp, fwd_exp, csd_exp_gamma_high
    # for baseline analyses
    filters_bas_gamma_high = make_dics(epo_bas.info,fwd_base,csd_bas_gamma_high,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=False)
    # clean memory
    del epo_bas, fwd_base, csd_bas_gamma_high
    # apply filters, calculate difference stcs & save
    # for experiment
    stc_ton_gamma_high, freqs_ton_gamma_high = apply_dics_csd(csd_ton_gamma_high,filters_exp_gamma_high)
    stc_neg_gamma_high, freqs_neg_gamma_high = apply_dics_csd(csd_neg_gamma_high,filters_exp_gamma_high)
    stc_pos_gamma_high, freqs_pos_gamma_high = apply_dics_csd(csd_pos_gamma_high,filters_exp_gamma_high)
    stc_diff_gamma_high = (stc_neg_gamma_high - stc_pos_gamma_high) / stc_ton_gamma_high
    stc_diff_gamma_high.save(fname=meg_dir+"nc_{}_stc_limb_mix_diff_gamma_high".format(meg))
    # clean memory
    del filters_exp_gamma_high, csd_neg_gamma_high, csd_pos_gamma_high
    # for baseline
    stc_ton_gamma_high, freqs_ton_gamma_high = apply_dics_csd(csd_ton_gamma_high,filters_bas_gamma_high)
    stc_rest_gamma_high, freqs_rest_gamma_high = apply_dics_csd(csd_rest_gamma_high,filters_bas_gamma_high)
    stc_tonbas_gamma_high = (stc_ton_gamma_high - stc_rest_gamma_high) / stc_rest_gamma_high
    stc_tonbas_gamma_high.save(fname=meg_dir+"nc_{}_stc_limb_mix_tonbas_gamma_high".format(meg))
    # clean memory
    del filters_bas_gamma_high, csd_rest_gamma_high, csd_ton_gamma_high
    del stc_rest_gamma_high, stc_ton_gamma_high, stc_neg_gamma_high, stc_pos_gamma_high
    # read mixed source space
    src = mne.read_source_spaces("{}nc_{}_limb_mix-src.fif".format(meg_dir,meg))
    labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
                   'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']
    # labels_parc = mne.read_labels_from_annot(mri, parc='aparc', subjects_dir=mri_dir)
    labels_dest = mne.read_labels_from_annot(mri, parc='aparc.a2009s', subjects_dir=mri_dir)
    # for each difference STC , extract a label_tc, get values out and sorted into group array
    # for experiment
    label_tc_gamma_high = mne.extract_label_time_course([stc_diff_gamma_high],labels_dest,src,mode='mean')
    X_label_gamma_high_diff.append(label_tc_gamma_high[0])
    del label_tc_gamma_high
    # for tone baseline
    label_tc_gamma_high_tonbas = mne.extract_label_time_course([stc_tonbas_gamma_high],labels_dest,src,mode='mean')
    X_label_gamma_high_tonbas.append(label_tc_gamma_high_tonbas[0])
    del label_tc_gamma_high_tonbas

# save the label_tc arrays for later analyses
label_tests = [X_label_gamma_high_diff,X_label_gamma_high_tonbas]
label_test_fname = ["X_label_gamma_high_diff","X_label_gamma_high_tonbas"]
for i,a in enumerate(label_tests):
    np.save(meg_dir+"{}".format(label_test_fname[i]),a)
