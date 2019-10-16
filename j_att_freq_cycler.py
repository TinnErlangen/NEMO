import mne
import matplotlib.pyplot as plt
import numpy as np

# browse through individual band power / condition

plt.ion()

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]

conds  = ["audio", "visual", "visselten", "rest", "zaehlen"]

filelistlist = []
for sub in subjs:
    filelist = []
    for cond in conds:
        filelist.append("{dir}nc_{sub}_{cond}-epo.fif".format(dir=proc_dir,sub=sub,cond=cond))
    filelistlist.append(filelist)
class Cycler():

    def __init__(self, filelist, conds, vmin=None, vmax=None):
        self.filelistlist = filelistlist
        self.conds = conds
        self.vmin = vmin
        self.vmax = vmax

    def go(self):
        self.fig, self.axes = plt.subplots(nrows=5,ncols=5)
        filelist = filelistlist.pop()
        for cond_idx, cond in enumerate(self.conds):
            filename = filelist.pop()
            epo = mne.read_epochs(filename)
            epo.plot_psd_topomap(axes=list(self.axes[cond_idx]),
            vmin=self.vmin,vmax=self.vmax, n_jobs=4, cmap="Reds")
        self.fig.tight_layout(pad=0,w_pad=0,h_pad=0)
cyc = Cycler(filelist, conds, vmin=None, vmax=None)
