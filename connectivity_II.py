import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
from mne.time_frequency import read_csd, pick_channels_csd
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
# freqs = {"gamma":(np.arange(35,56))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

exp_conds = ["exp","ton","neg","pos"]
bas_conds = ["rest","ton"]

# load the fsaverage ico4 source space
fs_src = mne.read_source_spaces("{}fsaverage_ico4-src.fif".format(meg_dir))

# now calculate connectivity for each subject
for meg,mri in sub_dict.items():
    print("Doing Connectivities for Subject: ",meg)

    # for the experimental conditions:
    print("Running Experimental Conditions")
    fwd_r = mne.read_forward_solution("{dir}nc_{meg}_from-fs_ico4_exp-r-fwd.fif".format(dir=meg_dir,meg=meg))
    # convert the forward model to one that defines two orthogonal dipoles at each source, that are tangential to a sphere
    fwd_tan = conpy.forward_to_tangential(fwd_r)
    # get pairs for connectivity calculation
    pairs = np.load("{}NEMO_ico4_connectivity_pairs.npy".format(meg_dir))
    # pairs are defined in fsaverage space, map them to the source space of the current subject
    fsaverage_to_subj = conpy.utils.get_morph_src_mapping(fs_src, fwd_tan['src'], indices=True, subjects_dir=mri_dir)[0]
    pairs = [[fsaverage_to_subj[v] for v in pairs[0]],
             [fsaverage_to_subj[v] for v in pairs[1]]]
    for cond in exp_conds:
        print("Calculations for Condition: ",cond)
        for freq,vals in freqs.items():
            print("Calculations for Frequency: ",freq)
            csd = read_csd("{dir}nc_{meg}-csd_{cond}_{freq}.h5".format(dir=meg_dir,meg=meg,cond=cond,freq=freq))
            csd = csd.mean()
            csd = pick_channels_csd(csd, fwd_tan['info']['ch_names'])
            con = conpy.dics_connectivity(vertex_pairs=pairs,fwd=fwd_tan,data_csd=csd,reg=0.05,n_jobs=8)
            con.save("{dir}nc_{meg}_{cond}_{freq}-connectivity.h5".format(dir=meg_dir,meg=meg,cond=cond,freq=freq))

    # for the baseline conditions:
    print("Running Baseline Conditions")
    fwd_r = mne.read_forward_solution("{dir}nc_{meg}_from-fs_ico4_bas-r-fwd.fif".format(dir=meg_dir,meg=meg))
    # convert the forward model to one that defines two orthogonal dipoles at each source, that are tangential to a sphere
    fwd_tan = conpy.forward_to_tangential(fwd_r)
    # get pairs for connectivity calculation
    pairs = np.load("{}NEMO_ico4_connectivity_pairs.npy".format(meg_dir))
    # pairs are defined in fsaverage space, map them to the source space of the current subject
    fsaverage_to_subj = conpy.utils.get_morph_src_mapping(fs_src, fwd_tan['src'], indices=True, subjects_dir=mri_dir)[0]
    pairs = [[fsaverage_to_subj[v] for v in pairs[0]],
             [fsaverage_to_subj[v] for v in pairs[1]]]
    for cond in bas_conds:
        print("Calculations for Condition: ",cond)
        for freq,vals in freqs.items():
            print("Calculations for Frequency: ",freq)
            csd = read_csd("{dir}nc_{meg}-csd_{cond}_{freq}.h5".format(dir=meg_dir,meg=meg,cond=cond,freq=freq))
            csd = csd.mean()
            csd = pick_channels_csd(csd, fwd_tan['info']['ch_names'])
            con = conpy.dics_connectivity(vertex_pairs=pairs,fwd=fwd_tan,data_csd=csd,reg=0.05,n_jobs=8)
            con.save("{dir}nc_{meg}_{cond}bas_{freq}-connectivity.h5".format(dir=meg_dir,meg=meg,cond=cond,freq=freq))
