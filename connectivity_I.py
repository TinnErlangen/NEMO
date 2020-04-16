import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
import conpy
from matplotlib import pyplot as plt
from mayavi import mlab

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_26":"ENR41"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
freqs = {"alpha":list(np.arange(8,14))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

# build common fsaverage ico4 source space to morph from, then back to later
fs_src = mne.setup_source_space('fsaverage',spacing='ico4',surface="white",subjects_dir=mri_dir,n_jobs=4)
fs_src.save("{}fsaverage_ico4-src.fif".format(meg_dir))

for meg,mri in sub_dict.items():
    # morph fsaverage ico4 source space to subject and save
    src = mne.morph_source_spaces(fs_src,mri,subjects_dir=mri_dir)
    src.save("{}nc_{}_from-fs_ico4-src.fif".format(meg_dir,meg))
    # create forward model and save
    # read trans file and BEM model that have been saved
    trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    bem = mne.read_bem_solution("{dir}nc_{meg}-bem.fif".format(dir=meg_dir,meg=meg))
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}nc_{sub}_1_ica-epo.fif".format(dir=meg_dir,sub=meg))
    ton = mne.read_epochs("{dir}nc_{sub}_2_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_a = mne.read_epochs("{dir}nc_{sub}_3_ica-epo.fif".format(dir=meg_dir,sub=meg))
    epo_b = mne.read_epochs("{dir}nc_{sub}_4_ica-epo.fif".format(dir=meg_dir,sub=meg))
    # build forward model from MRI and BEM  - for each experimental block
    fwd_rest = mne.make_forward_solution(rest.info, trans=trans, src=src, bem=bem,meg=True, mindist=0, n_jobs=8)
    fwd_ton = mne.make_forward_solution(ton.info, trans=trans, src=src, bem=bem,meg=True, mindist=0, n_jobs=8)
    fwd_a = mne.make_forward_solution(epo_a.info, trans=trans, src=src, bem=bem,meg=True, mindist=0, n_jobs=8)
    fwd_b = mne.make_forward_solution(epo_b.info, trans=trans, src=src, bem=bem,meg=True, mindist=0, n_jobs=8)
    # clear up memory
    del src, rest, ton, epo_a, epo_b
    # build averaged forward models for baseline comps (rest vs. ton) and experimental analyses (ton,exp blocks a+b)
    fwd_base = mne.average_forward_solutions([fwd_rest,fwd_ton], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_from-fs_ico4_base-fwd.fif".format(dir=meg_dir,meg=meg),fwd_base,overwrite=True)
    del fwd_rest
    fwd_exp = mne.average_forward_solutions([fwd_ton,fwd_a,fwd_b], weights=None)
    mne.write_forward_solution("{dir}nc_{meg}_from-fs_ico4_exp-fwd.fif".format(dir=meg_dir,meg=meg),fwd_exp,overwrite=True)
    del fwd_ton, fwd_a, fwd_b, fwd_exp

# now do the connectivity prep - select vertices and restrict forwards
max_sensor_dist = 0.08
ref_subject = "NEM_36"
fwd_ref = mne.read_forward_solution("{dir}nc_{meg}_from-fs_ico4_exp-fwd.fif".format(dir=meg_dir,meg=ref_subject))
fwd_ref = conpy.restrict_forward_to_sensor_range(fwd_ref, max_sensor_dist)

fwds_bas = [fwd_ref]
fwds_exp = [fwd_ref]
for meg,mri in sub_dict.items():
    fwds_bas.append(mne.read_forward_solution("{dir}nc_{meg}_from-fs_ico4_base-fwd.fif".format(dir=meg_dir,meg=meg)))
    fwds_exp.append(mne.read_forward_solution("{dir}nc_{meg}_from-fs_ico4_exp-fwd.fif".format(dir=meg_dir,meg=meg)))
vert_inds_bas = conpy.select_shared_vertices(fwds_bas, ref_src=fs_src,subjects_dir=mri_dir)
vert_inds_exp = conpy.select_shared_vertices(fwds_exp, ref_src=fs_src,subjects_dir=mri_dir)
del fwds_bas[0],fwds_exp[0]
del vert_inds_bas[0],vert_inds_exp[0]
for i,subj in enumerate(list(sub_dict.keys())):
    fwd_bas_r = conpy.restrict_forward_to_vertices(fwds_bas[i], vert_inds_bas[i])
    mne.write_forward_solution("{dir}nc_{meg}_from-fs_ico4_bas-r-fwd.fif".format(dir=meg_dir,meg=subj), fwd_bas_r,overwrite=True)
    fwd_exp_r = conpy.restrict_forward_to_vertices(fwds_exp[i], vert_inds_exp[i])
    mne.write_forward_solution("{dir}nc_{meg}_from-fs_ico4_exp-r-fwd.fif".format(dir=meg_dir,meg=subj), fwd_exp_r,overwrite=True)

# plot it - based on subject NEM_36 (last)
fig = mne.viz.plot_alignment(fwd_exp_r['info'],trans=trans,src=fwd_exp_r['src'], meg='sensors',surfaces='white',subjects_dir=mri_dir)
fig.scene.background = (1, 1, 1)  # white
g = fig.children[-1].children[0].children[0].glyph.glyph
g.scale_factor = 0.008
mlab.view(135, 120, 0.3, [0.01, 0.015, 0.058])

# choose the vertex pairs (based on distance in NEM_36) for which to compute connectivity
min_pair_dist = 0.03    # in meters
pairs = conpy.all_to_all_connectivity_pairs(fwd_exp_r, min_dist=min_pair_dist)
# store the pairs in fsaverage space
subj_to_fsaverage = conpy.utils.get_morph_src_mapping(fs_src, fwd_exp_r['src'], indices=True, subjects_dir=mri_dir)[1]
pairs = [[subj_to_fsaverage[v] for v in pairs[0]],
         [subj_to_fsaverage[v] for v in pairs[1]]]
np.save("{}NEMO_ico4_connectivity_pairs.npy".format(meg_dir), pairs)
