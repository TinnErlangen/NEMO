import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_16":"KIO12",
           "NEM_17":"DEN59","NEM_18":"SAG13","NEM_19":"ALC81","NEM_20":"PAG48",
           "NEM_21":"WKI71_fa","NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72","NEM_30":"DIU11","NEM_31":"BLE94",
           "NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa",
           "NEM_37":"EAM67"}
# sub_dict = {"NEM_24":"BII41"}
# freqs = {"theta":list(np.arange(4,6)),"alpha":list(np.arange(8,13)),"beta_low":list(np.arange(17,23)),
         # "beta_high":list(np.arange(26,34))}

for meg,mri in sub_dict.items():
    # # build source and BEM model from MRI
    # trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    # src = mne.setup_source_space(mri,surface="white",
    #                              subjects_dir=mri_dir,n_jobs=4)  ## uses 'oct6' as default, i.e. 4.9mm spacing appr.
    # src.save("{dir}nc_{meg}-src.fif".format(dir=meg_dir,meg=meg), overwrite=True)
    # bem_model = mne.make_bem_model(mri, subjects_dir=mri_dir, conductivity=[0.3])
    # bem = mne.make_bem_solution(bem_model)
    # mne.write_bem_solution("{dir}nc_{meg}-bem.fif".format(dir=meg_dir,meg=meg),bem)
    # mne.viz.plot_bem(subject=mri, subjects_dir=mri_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')
    # read source and BEM models that have been saved beamformer
    trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    src = mne.read_source_spaces("{dir}nc_{meg}-src.fif".format(dir=meg_dir,meg=meg))
    bem = mne.read_bem_solution("{dir}nc_{meg}-bem.fif".format(dir=meg_dir,meg=meg))
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # build forward model from MRI and BEM  - for each experimental block
    fwd_rest = mne.make_forward_solution(rest.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_1-fwd.fif".format(dir=meg_dir,meg=meg),fwd_rest,overwrite=True)
    fwd_ton = mne.make_forward_solution(ton.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_2-fwd.fif".format(dir=meg_dir,meg=meg),fwd_ton,overwrite=True)
    fwd_a = mne.make_forward_solution(epo_a.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_3-fwd.fif".format(dir=meg_dir,meg=meg),fwd_a,overwrite=True)
    fwd_b = mne.make_forward_solution(epo_b.info, trans=trans, src=src, bem=bem,meg=True, mindist=5.0, n_jobs=8)
    mne.write_forward_solution("{dir}nc_{meg}_4-fwd.fif".format(dir=meg_dir,meg=meg),fwd_b,overwrite=True)
    # build averaged forward models for baseline comps (rest vs. ton) and experimental analyses (ton,exp blocks a+b)
    fwd_base = mne.average_forward_solutions([fwd_rest,fwd_ton], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_base-fwd.fif".format(dir=meg_dir,meg=meg),fwd_base,overwrite=True)
    fwd_exp = mne.average_forward_solutions([fwd_ton,fwd_a,fwd_b], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_exp-fwd.fif".format(dir=meg_dir,meg=meg),fwd_exp,overwrite=True)
