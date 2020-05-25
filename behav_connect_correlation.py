import numpy as np
import mne
import pandas as pd
import random
from scipy import stats
from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

from sklearn.linear_model import LinearRegression
from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
import conpy
from mne.externals.h5io import write_hdf5
import corr_stats

meg_dir = "D:/NEMO_analyses/proc/"
mri_dir = "D:/freesurfer/subjects/"
base_dir = "D:/NEMO_analyses/behav/"
proc_dir = "D:/NEMO_analyses/proc/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_17":"DEN59","NEM_18":"SAG13",
           "NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72",
           "NEM_34":"KER27","NEM_36":"BRA52_fa","NEM_16":"KIO12","NEM_20":"PAG48","NEM_31":"BLE94","NEM_35":"MUN79"}
excluded_dict = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_26":"ENR41"}
subjs_all = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34",
        "NEM_35","NEM_36","NEM_37"]
excluded = ["NEM_19","NEM_21","NEM_30","NEM_32","NEM_33","NEM_37"]
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
#subjs = ["NEM_10","NEM_11"]

freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

# get the behavioral data array ready & choose the variable
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(proc_dir))
Behav = np.array(N_behav['Ton_Laut'])
cond = "Ton_Laut"
freqs = {"alpha":list(np.arange(8,14))}
save_dir = "D:/NEMO_analyses/plots/exp_behav/"

# load the fsaverage ico4 source space to morph back to
fs_src = mne.read_source_spaces("{}fsaverage_ico4-src.fif".format(meg_dir))

for freq,vals in freqs.items():
    diff_cons = []
    for meg,mri in sub_dict.items():
        con_neg = conpy.read_connectivity("{dir}nc_{meg}_neg_{freq}-connectivity.h5".format(dir=meg_dir,meg=meg,freq=freq))
        con_pos = conpy.read_connectivity("{dir}nc_{meg}_pos_{freq}-connectivity.h5".format(dir=meg_dir,meg=meg,freq=freq))
        con_diff = con_neg - con_pos
        # Morph the Connectivity back to the fsaverage brain.By now, the connection objects should define the same connection pairs between the same vertices.
        con_fsaverage = con_diff.to_original_src(fs_src, subjects_dir=mri_dir)
        diff_cons.append(con_fsaverage.data)
    diff_cons = np.array(diff_cons)

# get group diff average connectivity as data container
ga_con_diff = conpy.read_connectivity('{dir}NEMO_neg_vs_pos_contrast_{f}-avg-connectivity.h5'.format(dir=meg_dir,f=freq))

# setup for initial clustering
cluster_threshold = 2.845

# Perform a permutation test to only retain connections that are part of a significant bundle.
stats = corr_stats.cluster_permutation_test(diff_cons,Behav,cluster_threshold=cluster_threshold, src=fs_src, n_permutations=1000, verbose=True,
                                       alpha=0.05, n_jobs=2, seed=10, return_details=True, max_spread=0.01)
connection_indices, bundles, bundle_ts, bundle_ps, H0 = stats
con_clust = ga_con_diff[connection_indices]

# Save some details about the permutation stats to disk
write_hdf5('{dir}NEMO_N-P_connect_corr_{c}_{f}-stats.h5'.format(dir=meg_dir,c=cond,f=freq),dict(connection_indices=connection_indices,bundles=bundles,bundle_ts=bundle_ts,bundle_ps=bundle_ps,H0=H0),overwrite=True)

# Save the pruned grand average connection object
con_clust.save('{dir}NEMO_N-P_connect_corr_{c}_{f}-pruned-avg-connectivity.h5'.format(dir=meg_dir,c=cond,f=freq))
