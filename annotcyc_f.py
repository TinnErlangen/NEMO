#step 3 - visual inspection an marking of bad epochs

import mne
import matplotlib.pyplot as plt

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]

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
        self.epo.plot(n_epochs=12,n_channels=128) #these parameters work well for inspection of my 2sec epochs

    def plot(self,n_epochs=12,n_channels=128):
        self.epo.plot(n_epochs=n_epochs,n_channels=n_channels)

    def show_file(self):
        print("Current Epoch File: " + self.fn)

    def save(self):
        self.epo.save(self.fn[:-8]+'_m-epo.fif')
        if self.epo.info["bads"]:
            with open(self.fn[:-8]+'_badchans.txt', "w") as file:
                for b in self.epo.info["bads"]:
                    file.write(b+"\n")

cyc = Cycler(filelist)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file in the list for annotation and cyc.save() when done
#then cyc.go() again... until list is empty
