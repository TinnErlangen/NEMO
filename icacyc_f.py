import mne
import matplotlib.pyplot as plt

plt.ion()

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]
runs=["1"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(['{dir}{sub}_{run}_m-epo.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-ica.fif'.format(dir=proc_dir,sub=sub,run=run)])

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.epo = mne.read_epochs(self.fn[0])
        self.ica = mne.preprocessing.read_ica(self.fn[1])
        self.ica.plot_components()
        self.ica.plot_sources(self.epo)

    def plot_props(self,props):
        self.ica.plot_properties(self.epo,props)

    def show_file(self):
        print("Current raw file: " + self.fn[0])

    def save(self,comps=None):
        if comps:
            self.ica.apply(self.epo,exclude=comps).save(proc_dir+sub+"_"+run+"_ica-epo.fif")
        else:
            print("No components applied, saving anyway for consistency")
            self.epo.save(proc_dir+sub+"_"+run+"_ica-epo.fif")

cyc = Cycler(filelist)
