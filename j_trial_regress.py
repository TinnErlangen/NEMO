
import numpy as np
import mne
import matplotlib.pyplot as plt
plt.ion()

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import LinearRegression

from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
from mne.channels import find_ch_connectivity
from mne.datasets import limo
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne.viz import plot_topomap, plot_compare_evokeds, tight_layout
from mne import combine_evoked, find_layout
import pandas as pd

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

# parameters and setup
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "oct6"
conds = ["audio","visselten","visual"]
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
n_freqs = 1
n_srcs = 8196
n_subjs = len(subjs)
### permutation on betas
# number of random samples
perm_n = 100
# place holders for bootstrap samples
cluster_H0 = np.zeros(perm_n)
# setup connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",
                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)
del fs_src
connectivity = _setup_connectivity(cnx, n_srcs, n_freqs)
# threshold for clustering
threshold = dict(start=0, step=0.2)
#random_state = 42
random = np.random.RandomState()

df_laut = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")
df_laut["Intercept"] = 1
temp_df = []
for cond in conds:
    temp_df.append(df_laut[df_laut["Block"]==cond])
df_laut = pd.concat(temp_df)
predictor_vars = ["Laut","Block","Wav","Intercept","Subj"]
dm_laut = df_laut.copy()[predictor_vars]

df_ang["Intercept"] = 1
temp_df = []
for cond in conds:
    temp_df.append(df_ang[df_ang["Block"]==cond])
df_ang = pd.concat(temp_df)
predictor_vars = ["Angenehm","Intercept","Subj"] + conds + wavs + subjs
dm_ang = df_ang.copy()[predictor_vars]

dm = dm_ang

# # regression per subject
# betas = np.zeros((n_subjs,n_srcs*n_freqs))
# for sub_idx,sub in enumerate(subjs):
#     # make the df and data object for this particular subject
#     sub_dm = pd.DataFrame(columns=predictor_vars)
#     data = []
#     for cond_idx,cond in enumerate(conds):
#         for wav_idx,wav in enumerate(wavs):
#             data_temp = np.load("{dir}stcs/nc_{a}_{b}_{c}_{sp}_stc.npy".format(
#                                 dir=proc_dir,a=sub,b=cond,c=wav,sp=spacing))
#             for epo_idx in range(data_temp.shape[0]):
#                 sub_dm = sub_dm.append(dm[dm[cond]==1][dm[wav]==1][dm["Subj"]==sub])
#                 data.append(data_temp[epo_idx,])
#     sub_dm = sub_dm.drop("Subj",axis=1)
#     data = np.array(data)
#     Y = Vectorizer().fit_transform(data)
#     linear_model = LinearRegression(fit_intercept=False)
#     linear_model.fit(sub_dm, Y)
#     pred_col = predictor_vars.index('Angenehm')
#     coefs = get_coef(linear_model, 'coef_')
#     betas[sub_idx, :] = coefs[:, pred_col]
# beta = betas.mean(axis=0)

# regression on all data points
data = []
dm_new = pd.DataFrame(columns=predictor_vars)
idx_borders = []
idx_border = 0
for sub_idx,sub in enumerate(subjs):
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        idx_borders.append([idx_border])
        for wav_idx,wav in enumerate(wavs):
            data_temp = np.load("{dir}stcs/nc_{a}_{b}_{c}_{sp}_stc.npy".format(
                                dir=proc_dir,a=sub,b=cond,c=wav,sp=spacing))
            print(data_temp.shape)
            for epo_idx in range(data_temp.shape[0]):
                dm_new = dm_new.append(dm[dm[cond]==1][dm[wav]==1][dm["Subj"]==sub])
                data.append(data_temp[epo_idx,])
                idx_border += 1
        idx_borders[-1].append(idx_border)
data = np.array(data)
dm_new = dm_new.drop("Subj",axis=1)
for wav in wavs:
    dm_new = dm_new.drop(wav,axis=1)
Y = Vectorizer().fit_transform(data)
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(dm_new, Y)
pred_col = predictor_vars.index('Angenehm')
coefs = get_coef(linear_model, 'coef_')
betas = coefs[:, pred_col]
# find clusters
clusters, cluster_stats = _find_clusters(betas,threshold=threshold,
                                         connectivity=connectivity,
                                         tail=1)
for i in range(perm_n):
    resampled_data = data.copy()
    print("{} of {}".format(i,perm_n))
    # shuffle data within subject/condition
    for border in idx_borders:
        resample_inds = random.choice(range(border[0],border[1]),replace=False)
        resampled_data[border[0]:border[1],] = resampled_data[resample_inds,]
    Y = Vectorizer().fit_transform(resampled_data)
    linear_model.fit(dm_new, Y)
    coefs = get_coef(linear_model, 'coef_')
    betas = coefs[:, pred_col]
    # compute clustering on squared t-values (i.e., f-values)
    re_clusters, re_cluster_stats = _find_clusters(betas,threshold=threshold,
                                                   connectivity=connectivity,
                                                   tail=1)
    if len(re_clusters):
        cluster_H0[i] = re_cluster_stats.max()
    else:
        cluster_H0[i] = np.nan



# get upper CI bound from cluster mass H0
clust_threshold = np.quantile(cluster_H0[~np.isnan(cluster_H0)], [.95])

# good cluster inds
good_cluster_inds = np.where(cluster_stats > clust_threshold)[0]

# reshape clusters
#clusters = _reshape_clusters(clusters, (n_freqs, n_srcs))
