#step 3 - visual inspection and marking of bad epochs

import mne
import matplotlib.pyplot as plt

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16"]
#subjs = ["nc_NEM_25"]
runs = ["1","2","3","4"]
#runs = ["3"]

#collecting the files for annotation
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append('{dir}{sub}_{run}-epo.fif'.format(dir=proc_dir,sub=sub,run=run))

#definition of cycler object to go through the file list for annotation
class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.epo = mne.read_epochs(self.fn)
        self.epo.plot(n_epochs=10,n_channels=90,scalings=dict(mag=3e-12)) #these parameters work well for inspection of my 2sec epochs
        self.epo.plot_psd(fmax=50,average=False)

    def plot(self,n_epochs=10,n_channels=90):
        self.epo.plot(n_epochs=n_epochs,n_channels=n_channels,scalings=dict(mag=3e-12))

    def show_file(self):
        print("Current Epoch File: " + self.fn)

    def save(self):
        self.epo.save(self.fn[:-8]+'_m-epo.fif')
        if self.epo.info["bads"]:
            with open(self.fn[:-8]+'_badchans.txt', "w") as file:
                for b in self.epo.info["bads"]:
                    file.write(b+"\n")
        with open(self.fn[:-8]+'_epodrops.txt', "w") as file:
            for inx,d in enumerate(self.epo.drop_log):
                if d == ['USER']:
                    file.write("Epoch No. {inx} Condition {trig}\n".format(inx=inx+1,trig=self.epo.events[inx, 2]))

cyc = Cycler(filelist)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file in the list for annotation and cyc.save() when done
#then cyc.go() again... until list is empty
