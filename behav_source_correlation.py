import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

from sklearn.linear_model import LinearRegression

from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters

# setup files and folders, subject lists
## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

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

mri_suborder = [0,1,2,3,4,16,5,6,17,7,8,9,10,11,12,13,18,14,19,15]

freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

# so, first we gotta load a difference STC (N-P) array for every subject (!)
# that should be already the fsavg one, right? (should have same vertices)...
# that is the {sub}stc_fsavg_diff_{freq}.stc files ...
# an stc.data is array of n_dipoles x n_time ...
# that's why I collected the stc.data.T of every subject into a X_freq_diff array to do the cluster stats on ...
# however, I only saved that X_....npy for the gamma_high freq band and for all the label analyses..
# good thing though is that I can re-order the subjects first here, then collect the X together, and then save the X in the ordered order :)

# get the behavioral data array ready
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(proc_dir))

for freq,vals in freqs.items():

    # prepare the data arrays / objects needed
    all_diff_plot = []  # list for averaging and plotting group STC
    X_diff = []  #  list for collecting data for cluster stat analyses
    for sub in subjs:
        # load the STC data
        stc_fsavg_diff = mne.read_source_estimate("{dir}nc_{sub}_stc_fsavg_diff_{freq}".format(dir=meg_dir,sub=sub,freq=freq), subject='fsaverage')
        # collect the individual stcs into lists
        all_diff_plot.append(stc_fsavg_diff)
        X_diff.append(stc_fsavg_diff.data.T)
    # create group average stc for plotting later
    stc_sum = all_diff_plot.pop()
    for stc in all_diff_plot:
        stc_sum = stc_sum + stc
    NEM_all_stc_diff = stc_sum / len(subjs)
    # make data array for cluster permutation stats
    X_diff = np.array(X_diff)







# prepare connectivity for cluster stats
src = mne.read_source_spaces("{}fsaverage_ico5-src.fif".format(meg_dir))
connectivity = mne.spatial_src_connectivity(src)




# plot difference N-P in plain t-values on fsaverage
NEM_all_stc_diff_gamma_high.data = gh_t_obs.T
NEM_all_stc_diff_gamma_high.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(2,4,6)})
