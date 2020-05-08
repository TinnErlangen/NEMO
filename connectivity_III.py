import mne
import numpy as np
import conpy
from matplotlib import pyplot as plt
from mayavi import mlab
from mne.externals.h5io import write_hdf5

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
# freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

exp_conds = ["exp","ton","neg","pos"]
bas_conds = ["restbas","tonbas"]
exp_conds = ["pos","ton"]

# load the fsaverage ico4 source space to morph back to
fs_src = mne.read_source_spaces("{}fsaverage_ico4-src.fif".format(meg_dir))

# first make a group contrast of tone baseline vs. resting state; then experimental contrasts

# for each frequency band:
for freq,vals in freqs.items():
    # collect the connectivity objects into a dictionary by condition with a list each containing those of all subjects
    cons = dict()
    for cond in exp_conds:
        print('Reading connectivity for condition: ', freq, cond)
        cons[cond] = list()

        for meg,mri in sub_dict.items():
            con_subject = conpy.read_connectivity("{dir}nc_{meg}_{cond}_{freq}-connectivity.h5".format(dir=meg_dir,meg=meg,cond=cond,freq=freq))
            # Morph the Connectivity back to the fsaverage brain.By now, the connection objects should define the same connection pairs between the same vertices.
            con_fsaverage = con_subject.to_original_src(fs_src, subjects_dir=mri_dir)
            cons[cond].append(con_fsaverage)

    # Average the connection objects. To save memory, we add the data in-place.
    print('Averaging connectivity objects... ', freq)
    ga_con = dict()
    for cond in exp_conds:
        con = cons[cond][0].copy()
        for other_con in cons[cond][1:]:
            con += other_con
        con /= len(cons[cond])  # compute the mean
        ga_con[cond] = con
        con.save('{dir}NEMO_{c}_{f}-average-connectivity.h5'.format(dir=meg_dir,c=cond,f=freq))

    # Compute contrast between conditions
    contrast = ga_con[exp_conds[0]] - ga_con[exp_conds[1]]
    contrast.save('{dir}NEMO_pos_vs_ton_contrast_{f}-avg-connectivity.h5'.format(dir=meg_dir,f=freq))

    # Perform a permutation test to only retain connections that are part of a significant bundle.
    stats = conpy.cluster_permutation_test(cons['pos'], cons['ton'],cluster_threshold=5, src=fs_src, n_permutations=1000, verbose=True,
                                           alpha=0.05, n_jobs=2, seed=10, return_details=True, max_spread=0.01)
    connection_indices, bundles, bundle_ts, bundle_ps, H0 = stats
    con_clust = contrast[connection_indices]

    # Save some details about the permutation stats to disk
    write_hdf5('{dir}NEMO_pos_vs_ton_contrast_{f}-stats.h5'.format(dir=meg_dir,f=freq),dict(connection_indices=connection_indices,bundles=bundles,bundle_ts=bundle_ts,bundle_ps=bundle_ps,H0=H0),overwrite=True)

    # Save the pruned grand average connection object
    con_clust.save('{dir}NEMO_pos_vs_ton_contrast_{f}-pruned-avg-connectivity.h5'.format(dir=meg_dir,f=freq))

    # Summarize the connectivity in parcels
    labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s',subjects_dir=mri_dir)
    del labels[-2:]  # drop 'unknown-lh' label
    con_parc = con_clust.parcellate(labels, summary='degree',weight_by_degree=False)
    con_parc.save('{dir}NEMO_pos_vs_ton_contrast_{f}-pruned-label-avg-connectivity.h5'.format(dir=meg_dir,f=freq))
    # print them out
    print("The following conections remain for  ",freq)
    print(con_clust)
    print(con_parc)

print('[done]')
