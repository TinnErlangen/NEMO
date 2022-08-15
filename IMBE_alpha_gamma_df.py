import mne
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

proc_dir = "E:/v_MEG_DATALOG/NEMO_analyses_new/proc/"
proc_dir_b = "E:/v_MEG_DATALOG/NEMO_analyses/proc/"
trans_dir = "E:/v_MEG_DATALOG/NEMO_analyses/proc/trans_files/"
mri_dir = "E:/freesurfer/subjects/"
save_dir = "E:/v_MEG_DATALOG/NEMO_analyses/proc/IMBE/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}

# parameters
# (have to stick with old ones because filters etc already calced like that)
freqs_n = {"theta":list(np.arange(3,8)), "alpha":list(np.arange(8,14)), "beta_low":list(np.arange(14,22)),
           "beta_high":list(np.arange(22,31)), "gamma":list(np.arange(31,47))}
freqs_g = {"gamma_high":list(np.arange(65,96,2))}
cycs_n = {"theta":5, "alpha":7, "beta_low":9, "beta_high":11, "gamma":13}
cycs_g = {"gamma_high":15}
# lists for calcs below
freqs_n = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
cycs_n = [5, 5, 5, 5, 5, 7, 7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9,  9,  9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
freqs_g = [65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
cycs_g = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
fmins = [3, 8, 14, 22, 31]
fmaxs = [7, 13, 21, 30, 46]   # remember to use (65,95) as (fmin,fmax) for the gamma_high calculation

conds = {"rest": "rest", "ton": ["tonbas", "tonrat"]}

# DF_dictionary - for pandas frame of single trial data
# MERKE! brauche auch Gender table zum Zuordnen für Subject ... Column nachtragen! #
keys1 = ["Subject", "Cond"]
# 162 is number of vertices per hemisphere in ico3 SRCs, i.e. target vertices for power values;
# power variables are Vertex number + hemisphere + frequency band
keys2 = ["V_{}_{}_{}".format(i+1, h, f) for f in ("alpha", "gamma") for h in ("l", "r") for i in range(162)]
DF_dict = {k: [] for k in keys1 + keys2}

# load fsaverage src for morphing
fs_src = mne.read_source_spaces("{}fsaverage_ico2-src.fif".format(save_dir))

## POWER ROI CALCULATIONS
for meg,mri in sub_dict.items():
    # load data, filters for DICS beamformer, make source morph for fsaverage target
    epo_all = mne.read_epochs("{}{}-epo.fif".format(proc_dir,meg))
    # # get some old mixed stc, to get surface only and compute subject-to-fsaverage morph
    # stc = mne.read_source_estimate("{}nc_{}_stc_restbas_alpha".format(proc_dir_b, meg))
    # stc = stc.surface()
    # morph = mne.compute_source_morph(stc, subject_from=mri, subject_to="fsaverage",
    #                                  subjects_dir=mri_dir, src_to=fs_src)
    # del stc
    filters_n = mne.beamformer.read_beamformer('{}{}-dics.h5'.format(proc_dir, meg))
    filters_g = mne.beamformer.read_beamformer('{}{}-gamma-dics.h5'.format(proc_dir, meg))
    # calc single trial data for the conditions
    for cond, clab in conds.items():
        # select condition epochs
        epos = epo_all[clab]
        # calc loop
        for i in range(len(epos)):
            epo = epos[i]
            # calculate csd and csd_gamma
            csd_n = mne.time_frequency.csd_morlet(epo, frequencies=freqs_n, n_jobs=8, n_cycles=cycs_n, decim=1)
            csd_g = mne.time_frequency.csd_morlet(epo, frequencies=freqs_g, n_jobs=8, n_cycles=cycs_g, decim=1)
            # apply filters
            stc, frqs = mne.beamformer.apply_dics_csd(csd_n.mean(fmins,fmaxs),filters_n)
            del csd_n
            stc_a = stc.copy()
            stc_a.data = np.expand_dims(stc.data[:,1], axis=1)      # get alpha band only
            stc_a = stc_a.surface()                                 # get surface data only
            morph = mne.compute_source_morph(stc_a, subject_from=mri, subject_to="fsaverage",
                                             subjects_dir=mri_dir, src_to=fs_src)
            stc_fs_a = morph.apply(stc_a)
            del stc_a
            stc_g, frqs_g = mne.beamformer.apply_dics_csd(csd_g.mean(65,95),filters_g)
            del csd_g
            stc_g = stc_g.surface()                                 # get surface data only
            stc_fs_g = morph.apply(stc_g)
            del stc_g
            # fill the values into DF_dict
            DF_dict["Subject"].append(meg)
            DF_dict["Cond"].append(cond)
            # vertex numbers in rh corr. to second half of indices in stc.data
            for i in range(162):
                DF_dict["V_{}_l_alpha".format(i+1)].append(stc_fs_a.data[i][0])
                DF_dict["V_{}_r_alpha".format(i+1)].append(stc_fs_a.data[i+162][0])
                DF_dict["V_{}_l_gamma".format(i+1)].append(stc_fs_g.data[i][0])
                DF_dict["V_{}_r_gamma".format(i+1)].append(stc_fs_g.data[i+162][0])
            del stc_fs_a, stc_fs_g

DF_imbe = pd.DataFrame(DF_dict)
DF_imbe.dtypes()
DF_imbe.to_csv(save_dir + "Ton_alpha_gamma.csv")
# convert the power values for stats: multiply by e+30, to make fT^2 out of T^2 & then take the log10 to make linear; power columns are [2:]
DF_imbe.iloc[:, 2:] = DF_imbe.iloc[:, 2:].mul(1e+30)
DF_imbe.iloc[:, 2:] = np.log10(DF_imbe.iloc[:, 2:])
