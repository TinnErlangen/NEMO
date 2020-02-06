import mne
import numpy as np

base_dir = "D:/NEMO_analyses/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_37"]
#subjs = ["nc_NEM_25"]

with open(base_dir+"NEMO_condnums_all.txt","w") as file:
    file.write("Subject\tRest\tTonbas\tNeg\tPos\tTon_r1\tTon_r2\tTon_s1\tTon_s2\tNeg_r1\tNeg_r2\tNeg_s1\tNeg_s2\tPos_r1\tPos_r2\tPos_s1\tPos_s2\n")
    for subj in subjs:
        epo1 = mne.read_epochs(proc_dir+subj+"_1_ica-epo.fif")
        epo2 = mne.read_epochs(proc_dir+subj+"_2_ica-epo.fif")
        epo3 = mne.read_epochs(proc_dir+subj+"_3_ica-epo.fif")
        epo4 = mne.read_epochs(proc_dir+subj+"_4_ica-epo.fif")
        file.write("{s}\t{rest}\t{tonbas}\t{neg}\t{pos}\t{tr1}\t{tr2}\t{ts1}\t{ts2}\t{nr1}\t{nr2}\t{ns1}\t{ns2}\t{pr1}\t{pr2}\t{ps1}\t{ps2}\n".format(
                   s=subj,rest=len(epo1),tonbas=len(epo2),neg=len(epo3['negative'])+len(epo4['negative']),pos=len(epo3['positive'])+len(epo4['positive']),
                   tr1=len(epo2['ton_r1']),tr2=len(epo2['ton_r2']),ts1=len(epo2['ton_s1']),ts2=len(epo2['ton_s2']),
                   nr1=len(epo3['negative/r1'])+len(epo4['negative/r1']),nr2=len(epo3['negative/r2'])+len(epo4['negative/r2']),
                   ns1=len(epo3['negative/s1'])+len(epo4['negative/s1']),ns2=len(epo3['negative/s2'])+len(epo4['negative/s2']),
                   pr1=len(epo3['positive/r1'])+len(epo4['positive/r1']),pr2=len(epo3['positive/r2'])+len(epo4['positive/r2']),
                   ps1=len(epo3['positive/s1'])+len(epo4['positive/s1']),ps2=len(epo3['positive/s2'])+len(epo4['positive/s2'])))
