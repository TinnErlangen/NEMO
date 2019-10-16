import mne
import numpy as np

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_37"]
#subjs = ["nc_NEM_25"]
runs = ["3","4"]
#runs=["1","2"]
#runs = ["4"]

with open(base_dir+"NEMO_condnums.txt","w") as file:
    file.write("Subject\tRun\tNeg\tPos\tn/r1\tn/r2\tn/s1\tn/s2\tp/r1\tp/r2\tp/s1\tp/s2\n")
    for subj in subjs:
        for run in runs:
            epo = mne.read_epochs(proc_dir+subj+"_"+run+"_ica-epo.fif")
            file.write("{s}\t{r}\t{n}\t{p}\t{nr1}\t{nr2}\t{ns1}\t{ns2}\t{pr1}\t{pr2}\t{ps1}\t{ps2}\n".format(
                       s=subj,r=run,n=len(epo['negative']),p=len(epo['positive']),nr1=len(epo['negative/r1']),
                       nr2=len(epo['negative/r2']),ns1=len(epo['negative/s1']),ns2=len(epo['negative/s2']),
                       pr1=len(epo['positive/r1']),pr2=len(epo['positive/r2']),ps1=len(epo['positive/s1']),
                       ps2=len(epo['positive/s2'])))
