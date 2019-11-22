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
         "beta_high":list(np.arange(26,35))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30}

for meg,mri in sub_dict.items():
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # separate the experimental conditions in the MEG data
    # neg_a = epo_a["negative"]
    # neg_b = epo_b["negative"]
    # pos_a = epo_a["positive"]
    # pos_b = epo_b["positive"]
    # load the forward models from each experimental block
    fwd_rest = mne.read_forward_solution("{dir}nc_{meg}_1-fwd.fif".format(dir=meg_dir,meg=meg))
    fwd_ton =  mne.read_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg))
    # fwd_a = mne.read_forward_solution("{dir}nc_{meg}_3-fwd.fif".format(dir=meg_dir,meg=meg))
    # fwd_b = mne.read_forward_solution("{dir}nc_{meg}_4-fwd.fif".format(dir=meg_dir,meg=meg))
    # load CSD matrix for each block and conditions
    csd_rest = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_rest_gamma.h5".format(dir=meg_dir,meg=meg))
    csd_ton = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_ton_gamma.h5".format(dir=meg_dir,meg=meg))
    # csd_a = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_a.h5".format(dir=meg_dir,meg=meg))
    # csd_b = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_b.h5".format(dir=meg_dir,meg=meg))
    # csd_pos_a = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_a.h5".format(dir=meg_dir,meg=meg))
    # csd_pos_b = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_pos_b.h5".format(dir=meg_dir,meg=meg))
    # csd_neg_a = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_a.h5".format(dir=meg_dir,meg=meg))
    # csd_neg_b = mne.time_frequency.read_csd("{dir}nc_{meg}-csd_neg_b.h5".format(dir=meg_dir,meg=meg))

    # make DICS beamformers / filters for each block
    ## template: filters_a = make_dics(epo_a.info,fwd_a,csd_a,pick_ori='max-power',rank='full',inversion='single',weight_norm="unit-noise-gain",normalize_fwd=False,real_filter=True)
    filters_rest = make_dics(rest.info,fwd_rest,csd_rest,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=True)
    filters_ton = make_dics(ton.info,fwd_ton,csd_ton,pick_ori='max-power',rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=True)
    # filters_a = make_dics(epo_a.info,fwd_a,csd_a,pick_ori=None,rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=True)
    # filters_b = make_dics(epo_b.info,fwd_b,csd_b,pick_ori=None,rank=None,inversion='single',weight_norm=None,normalize_fwd=True,real_filter=True)

    # calculate Source Estimates for the experimental conditions in each block
    stc_rest, freqs_rest = apply_dics_csd(csd_rest.mean(),filters_rest)
    stc_ton, freqs_ton = apply_dics_csd(csd_ton.mean(),filters_ton)
    # stc_pos_a, freqs_pos_a = apply_dics_csd(csd_pos_a.mean(),filters_a) # in mne tutorial they use csd.mean() for the whole freq band
    # stc_pos_b, freqs_pos_b = apply_dics_csd(csd_pos_b.mean(),filters_b)
    # stc_neg_a, freqs_neg_a = apply_dics_csd(csd_neg_a.mean(),filters_a) # in mne tutorial they use csd.mean() for the whole freq band
    # stc_neg_b, freqs_neg_b = apply_dics_csd(csd_neg_b.mean(),filters_b)
    # stc_pos = (stc_pos_a + stc_pos_b) / 2
    # stc_neg = (stc_neg_a + stc_neg_b) / 2
    stc_tonloc = (stc_ton - stc_rest) / stc_rest
    # stc_diff = (stc_neg - stc_pos) / stc_rest

    # plot the difference between conditions
    stc_tonloc.plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)
    # stc_diff.plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)

    # # plot tones vs. resting state; pos/neg vs. tones
    # ton_v_rest = stc_ton - stc_rest
    # ton_v_rest.mean().plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)
    # pos_v_ton = stc_pos - stc_ton
    # pos_v_ton.mean().plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)
    # neg_v_ton = stc_neg - stc_ton
    # neg_v_ton.mean().plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)
