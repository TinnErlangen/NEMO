import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet,csd_multitaper
import numpy as np
from scipy import stats

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_17":"DEN59","NEM_18":"SAG13",
           "NEM_22":"EAM11","NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41",
           "NEM_27":"HIU14","NEM_28":"WAL70","NEM_29":"KIL72",
           "NEM_34":"KER27","NEM_36":"BRA52_fa","NEM_16":"KIO12","NEM_20":"PAG48","NEM_31":"BLE94","NEM_35":"MUN79"}
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23",}

# read labels for catching their names
labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
               'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']
labels_dest = mne.read_labels_from_annot('GIZ04', parc='aparc.a2009s', subjects_dir=mri_dir)

label_test_fname = ["X_label_gamma_high_diff","X_label_gamma_high_tonbas"]

# # loop through tests to be done, make permutation t-test over labels, spit out significant ones
# for i, test in enumerate(label_test_fname):
#     X = np.load(meg_dir+"{}.npy".format(test))
#     X = np.squeeze(X)
#     t_obs, pvals, H0 = mne.stats.permutation_t_test(X, n_permutations=1024, tail=0, n_jobs=4, seed=None) # first time done with 10000 permutations
#     good_pval_inds = np.where(pvals < 0.05)[0]
#     print("{t} significant in label no.s:{n}".format(t=test,n=good_pval_inds))
#     print("With t Values: {}, and p Values: {}".format(t_obs[good_pval_inds],pvals[good_pval_inds]))
#     print("Significant label names are:")
#     for lab_i in good_pval_inds:
#         if lab_i < 150:
#             print("{}".format(labels_dest[lab_i]))
#         else:
#             lab_i = lab_i - 150
#             print("{}".format(labels_limb[lab_i]))
#     next_pval_inds = np.where(pvals < 0.1)[0]
#     for lab_i in next_pval_inds:
#         if lab_i not in good_pval_inds:
#             print("marginally sign. label {no} with T={t} and P={p}".format(no=lab_i,t=t_obs[lab_i],p=pvals[lab_i]))
#             if lab_i < 150:
#                 print("{}".format(labels_dest[lab_i]))
#             else:
#                 lab_i = lab_i - 150
#                 print("{}".format(labels_limb[lab_i]))

# loop through tests to be done, make permutation t-test over labels, spit out significant ones - this time for selected ROIs only
roi_labs = [155,16,34,92,94,96,6,46,122,124,26,150,64,146,80,70,50,108,48,110,161,17,35,93,95,97,7,47,123,125,27,156,65,147,81,71,51,109,49,111]
lab_names = []
for l in labels_dest:
    lab_names.append(l.name)
for l in labels_limb:
    lab_names.append(l)

filename = "D:/NEMO_analyses/roi_gamma_high_permutation.txt"
with open(filename,"w") as file:

    for i, test in enumerate(label_test_fname):
        X = np.load(meg_dir+"{}.npy".format(test))
        X = np.squeeze(X)
        X_roi = []
        for l in roi_labs:
            X_roi.append(X[:,l])
        X_roi = np.array(X_roi)
        X_roi = X_roi.T
        t_obs, pvals, H0 = mne.stats.permutation_t_test(X_roi, n_permutations=1024, tail=0, n_jobs=4, seed=None)
        good_pval_inds = np.where(pvals < 0.05)[0]
        file.write("{t} significant in label no.s:{n}\n".format(t=test,n=good_pval_inds))
        file.write("With t Values: {}, and p Values: {}\n".format(t_obs[good_pval_inds],pvals[good_pval_inds]))
        file.write("Significant label names are:\n")
        for lab_i in good_pval_inds:
            lab_n = roi_labs[lab_i]
            if lab_n < 150:
                file.write("{}".format(labels_dest[lab_n].name))
            else:
                lab_n = lab_n - 150
                file.write("{}\n".format(labels_limb[lab_n]))
        file.write(" TEST - {}\n".format(test))
        file.write("Index\tNumber\tLabel\tT-Value\tp-Value\n")
        for i in range(len(t_obs)):
            n = roi_labs[i]
            file.write("{i}\t{n}\t{l}\t{t}\t{p}\n".format(i=i,n=roi_labs[i],l=lab_names[n],t=t_obs[i],p=pvals[i]))

# get everyones values for the amygdala
filename = "D:/NEMO_analyses/amygdala_indiv_diff_gamma_high_values.txt"
with open(filename,"w") as file:

    for i, test in enumerate(label_test_fname):
        X = np.load(meg_dir+"{}.npy".format(test))
        X = np.squeeze(X)
        file.write(" TEST - {}\n".format(test))
        A_lh = X[:,155]
        A_rh = X[:,161]
        file.write("Left Amygdala:\n")
        for v in A_lh:
            file.write("{}\t".format(v))
        file.write("\nMean\tSD\n")
        file.write("{m}\t{s}\n".format(m=stats.tmean(A_lh),s=stats.tstd(A_lh)))
        file.write("\nRight Amygdala:\n")
        for v in A_rh:
            file.write("{}\t".format(v))
        file.write("\nMean\tSD\n")
        file.write("{m}\t{s}\n".format(m=stats.tmean(A_rh),s=stats.tstd(A_rh)))
