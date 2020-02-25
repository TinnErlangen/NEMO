## preparation of source space and forward solutions as mixed source space with cortex and subcortical volume structures

import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48",
           "NEM_21":"WKI71_fa","NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72","NEM_30":"DIU11","NEM_31":"BLE94",
           "NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa",
           "NEM_37":"EAM67"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_16":"KIO12",}
# freqs = {"theta":list(np.arange(4,6)),"alpha":list(np.arange(8,13)),"beta_low":list(np.arange(17,23)),
         # "beta_high":list(np.arange(26,34))}

for meg,mri in sub_dict.items():
    # build an 'ico5' surface source space & save it
    src = mne.setup_source_space(mri,surface="white",subjects_dir=mri_dir,spacing='ico5',n_jobs=4)  ## uses 'oct6' as default, here using 'ico5' to match with fsaverage for morphing etc.
    src.save("{dir}nc_{meg}_ico5-src.fif".format(dir=meg_dir,meg=meg), overwrite=True)
    # read trans file and BEM model that have been saved
    trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    bem = mne.read_bem_solution("{dir}nc_{meg}-bem.fif".format(dir=meg_dir,meg=meg))
    # build mixed source model from MRI - surface and volume(from label list and aseg.mgz file) & save it
    labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
                  'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']
    limb_src = mne.setup_volume_source_space(mri, mri=mri_dir+"/fsaverage/mri/aseg.mgz", pos=5.0, bem=bem,
                                            volume_label=labels_limb, subjects_dir=mri_dir,add_interpolator=True,verbose=True)
    src += limb_src
    # print out the number of spaces and points
    n = sum(src[i]['nuse'] for i in range(len(src)))
    print('the src space contains %d spaces and %d points' % (len(src), n))
    # save the mixed source space
    src.save(meg_dir+"nc_{}_limb_mix-src.fif".format(meg), overwrite=True)
    # clear memory
    del limb_src
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # build forward model from MRI and BEM  - for each experimental block
    fwd_rest = mne.make_forward_solution(rest.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=4)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_1-fwd.fif".format(dir=meg_dir,meg=meg),fwd_rest,overwrite=True)
    fwd_ton = mne.make_forward_solution(ton.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=4)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_2-fwd.fif".format(dir=meg_dir,meg=meg),fwd_ton,overwrite=True)
    fwd_a = mne.make_forward_solution(epo_a.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=4)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_3-fwd.fif".format(dir=meg_dir,meg=meg),fwd_a,overwrite=True)
    fwd_b = mne.make_forward_solution(epo_b.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=4)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_4-fwd.fif".format(dir=meg_dir,meg=meg),fwd_b,overwrite=True)
    # clear up memory
    del src, rest, ton, epo_a, epo_b
    # build averaged forward models for baseline comps (rest vs. ton) and experimental analyses (ton,exp blocks a+b)
    fwd_base = mne.average_forward_solutions([fwd_rest,fwd_ton], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_base-fwd.fif".format(dir=meg_dir,meg=meg),fwd_base,overwrite=True)
    del fwd_rest
    fwd_exp = mne.average_forward_solutions([fwd_ton,fwd_a,fwd_b], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_limb_mix_exp-fwd.fif".format(dir=meg_dir,meg=meg),fwd_exp,overwrite=True)
