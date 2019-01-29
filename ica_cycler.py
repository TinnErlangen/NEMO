import mne
import matplotlib.pyplot as plt

plt.ion()

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10"]
runs = [str(x+1) for x in range(5)]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(["{dir}nc_{sub}_{run}_hand-raw.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_hand-ica.fif".format(dir=proc_dir,sub=sub,run=run)])

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        self.ica = mne.preprocessing.read_ica(self.fn[1])
        self.ica.plot_components()
        self.ica.plot_sources(self.raw)

    def plot_props(self,props):
        self.ica.plot_properties(self.raw,props)

    def show_file(self):
        print("Current raw file: "+self.fn[0])

    def save(self,comps=None):
        if comps:
            self.ica.apply(self.raw,exclude=comps).save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)
        else:
            print("No components applied, saving anyway for consistency.")
            self.raw.save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)

cyc = Cycler(filelist)
