import mne
import matplotlib.pyplot as plt

plt.ion()

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append('{dir}{sub}_{run}-epo.fif'.format(dir=proc_dir,sub=sub,run=run))

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.epo = mne.read_epochs(self.fn)
        self.epo.plot(n_epochs=12,n_channels=128)

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
