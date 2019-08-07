import mne
import numpy as np
import pickle

def compensate(raw,weights=None,direction=1):
    if not raw.preload:
        raw.load_data()
    if not weights:
        from sklearn.linear_model import LinearRegression
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
        estimator = LinearRegression(normalize=True)
        Y_pred = estimator.fit(
        raw[ref_picks][0].T,raw[meg_picks][0].T).predict(raw[ref_picks][0].T)
        raw._data[meg_picks] -= direction*Y_pred.T
    else:
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False,exclude=[])
        ref_picks = mne.pick_channels(raw.ch_names,weights["comp_names"])
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
        # build matrix, rows and columns don't correspond at first
        comp_names = weights["comp_names"]
        comp_mat = np.zeros((weights["chan_num"],weights["comp_num"]))
        weights = weights["weights"]
        for ch_idx,ch in zip(meg_picks,meg_ch_names):
            comp_mat[ch_idx,:] = np.array(weights[ch])
        megdata = np.dot(comp_mat,raw._data[ref_picks,])
        raw._data[meg_picks,] -= direction*megdata

# do an ICA decomposition on data, and reference channels.

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["ATT_16","ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
runs = [str(x+1) for x in range(5)]
#runs = ["1"]

base_dir ="../"
proc_dir = base_dir+"proc/"

with open("/home/jeff/reftest/bin/compsup1","rb") as f:
    digital_comp = pickle.load(f)["digital"]

for sub in subjs:
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_p_hand-raw.fif".format(
        dir=proc_dir,sub=sub,run=run),preload=True)

        # compensate(raw,digital_comp,direction=-1)
        # compensate(raw)
        # raw.info["bads"] += ["MRyA", "MRyaA"]
        # raw.save("{dir}nc_{sub}_{run}_p_hand_C-raw.fif".format(
        #     dir=proc_dir,sub=sub,run=run),overwrite=True)

        icaref = mne.preprocessing.ICA(n_components=6,max_iter=10000,
                                       method="picard",allow_ref_meg=True)
        picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        icaref.fit(raw,picks=picks)
        icaref.save("{dir}nc_{sub}_{run}_p_hand_ref-ica.fif".format(dir=proc_dir,
                                                                  sub=sub,
                                                                  run=run))

        icameg = mne.preprocessing.ICA(n_components=100,max_iter=10000,
                                       method="picard")
        picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        icameg.fit(raw,picks=picks)
        icameg.save("{dir}nc_{sub}_{run}_p_hand_meg-ica.fif".format(dir=proc_dir,
                                                                  sub=sub,
                                                                  run=run))

        ica = mne.preprocessing.ICA(n_components=100,max_iter=10000,
                                       method="picard",allow_ref_meg=True)
        picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
        ica.fit(raw,picks=picks)
        ica.save("{dir}nc_{sub}_{run}_p_hand-ica.fif".format(dir=proc_dir,
                                                                  sub=sub,
                                                                  run=run))
