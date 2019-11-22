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
sub_dict = {"NEM_24":"BII41"}
freqs = {"theta":list(np.arange(4,6)),"alpha":list(np.arange(8,13)),"beta_low":list(np.arange(17,23)),
         "beta_high":list(np.arange(26,34))}

for meg,mri in sub_dict.items():
    # build forward model from MRI and BEM
    trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    src = mne.setup_source_space(mri,surface="white",
                                 subjects_dir=mri_dir,n_jobs=4)
    src.save("{dir}nc_{meg}-src.fif".format(dir=meg_dir,meg=meg), overwrite=True)
    bem_model = mne.make_bem_model(mri, subjects_dir=mri_dir, conductivity=[0.3])
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{dir}nc_{meg}-bem.fif".format(dir=meg_dir,meg=meg),bem)
    mne.viz.plot_bem(subject=mri, subjects_dir=mri_dir,
                 brain_surfaces='white', src=src, orientation='coronal')
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    #interpolate bad n_channels
    epo_a.interpolate_bads()
    epo_b.interpolate_bads()
    fwd_a = mne.make_forward_solution(epo_a.info, trans=trans, src=src, bem=bem,meg=True, mindist=1.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_3-fwd.fif".format(dir=meg_dir,meg=meg),fwd_a,overwrite=True)
    fwd_b = mne.make_forward_solution(epo_b.info, trans=trans, src=src, bem=bem,meg=True, mindist=1.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_4-fwd.fif".format(dir=meg_dir,meg=meg),fwd_b,overwrite=True)

    # calculate DICS filters across all conditions
    # force-append epo_a+b only for calc csd here
        #override coil positions of block b with those of block a for concatenation (sensor level only!!)
    epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
        #concatenate data of both blocks
    epo = mne.concatenate_epochs([epo_a,epo_b])
    # calculate DICS filters across all conditions
    csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1)
    #csd = csd_multitaper(epo,fmin=7,fmax=14,bandwidth=1,adaptive=True)
    #csd = csd_morlet(epo, frequencies=freqs, n_jobs=8, n_cycles=7, decim=3)
    #filters_a = make_dics(epo_a.info,fwd_a,csd,pick_ori='max power',rank='full',weight_norm="unit-noise-gain",normalize_fwd=False,real_filter=True)
    filters_a = make_dics(epo_a.info,fwd_a,csd,pick_ori=None,rank='full',weight_norm=None,normalize_fwd=False,real_filter=False)
    filters_b = make_dics(epo_b.info,fwd_b,csd,pick_ori=None,rank='full',weight_norm=None,normalize_fwd=False,real_filter=False)
    neg_a = epo_a["negative"]
    neg_b = epo_b["negative"]
    pos_a = epo_a["positive"]
    pos_b = epo_b["positive"]
    # csd_pos_a = csd_morlet(pos, frequencies=freqs, n_jobs=8, n_cycles=9, decim=3)
    # csd_neg_a = csd_morlet(neg, frequencies=freqs, n_jobs=8, n_cycles=9, decim=3)
    csd_pos_a = csd_multitaper(pos_a,fmin=7,fmax=14,bandwidth=1)
    stc_pos_a, freqs_pos_a = apply_dics_csd(csd_pos_a,filters_a) # in mne tutorial they use csd.mean() for the whole freq band
    csd_pos_b = csd_multitaper(pos_b,fmin=7,fmax=14,bandwidth=1)
    stc_pos_b, freqs_pos_b = apply_dics_csd(csd_pos_b,filters_b)
    csd_neg_a = csd_multitaper(neg_a,fmin=7,fmax=14,bandwidth=1)
    stc_neg_a, freqs_neg_a = apply_dics_csd(csd_neg_a,filters_a) # in mne tutorial they use csd.mean() for the whole freq band
    csd_neg_b = csd_multitaper(neg_b,fmin=7,fmax=14,bandwidth=1)
    stc_neg_b, freqs_neg_b = apply_dics_csd(csd_neg_b,filters_b)
    stc_pos = (stc_pos_a + stc_pos_b) /2
    stc_neg = (stc_neg_a + stc_neg_b) /2
    stc_diff = stc_neg - stc_pos
    # plot the difference between conditions
    stc_diff.plot(subjects_dir=mri_dir,subject=mri,hemi='split',time_viewer=True)
