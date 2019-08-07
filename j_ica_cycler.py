import mne
import matplotlib.pyplot as plt
import numpy as np
import pickle

# reduces the pain of finding and removing ICA components. Analogous to
# annot_cycler.py

plt.ion()

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["ATT_11"]
runs = [str(x+1) for x in range(5)]
runs = ["3"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(["{dir}nc_{sub}_{run}_p_hand-raw.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_p_hand_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_p_hand_meg-ica.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_p_hand-ica.fif".format(dir=proc_dir,sub=sub,run=run)])

ref_comp_num = 20

class Cycler():

    def __init__(self,filelist,ref_comp_num):
        self.filelist = filelist
        self.ref_comp_num = ref_comp_num

    def go(self,idx=0):
        plt.close('all')
        # load the next raw/ICA files
        self.fn = self.filelist.pop(idx)
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        self.icaref = mne.preprocessing.read_ica(self.fn[1])
        self.icameg = mne.preprocessing.read_ica(self.fn[2])
        self.ica = mne.preprocessing.read_ica(self.fn[3])

        #housekeeping on reference components, add them to raw data
        refcomps = self.icaref.get_sources(self.raw)
        for c in refcomps.ch_names[:self.ref_comp_num]: # they need to have REF_ prefix to be recognised by MNE algorithm
            refcomps.rename_channels({c:"REF_"+c})
        self.raw.add_channels([refcomps])

        self.comps = []

        # plot everything out for overview
        self.ica.plot_components(picks=list(range(20)))
        self.ica.plot_sources(self.raw)
        # self.icameg.plot_sources(self.raw)
        # self.icaref.plot_sources(self.raw, picks = list(range(self.ref_comp_num)))
        # self.raw.plot(n_channels=64,duration=120,scalings="auto")
        self.raw.plot_psd(fmax=40)

    def plot_props(self,props=None):
        # in case you want to take a closer look at a component
        if not props:
            props = self.comps
        self.ica.plot_properties(self.raw,props)

    def show_file(self):
        print("Current raw file: "+self.fn[0])

    def without(self,comps=None,fmax=40):
        # see what the data would look like if we took comps out
        self.comps += self.ica.exclude
        if not comps:
            comps = self.comps
        test = self.raw.copy()
        test.load_data()
        test = self.ica.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax)
        test.plot(duration=30,n_channels=30)
        self.test = test

    def identify_bad(self,method,threshold=0.5):
        # search for components which correlate with noise
        if isinstance(method,str):
            method = [method]
        elif not isinstance(method,list):
            raise ValueError('"method" must be string or list.')
        for meth in method:
            print(meth)
            if meth == "eog":
                inds, scores = self.ica.find_bads_eog(self.raw)
            elif meth == "ecg":
                inds, scores = self.ica.find_bads_ecg(self.raw)
            elif meth == "ref":
                inds, scores = self.ica.find_bads_ref(self.raw, method="separate",
                                                      threshold=threshold,
                                                      bad_measure="cor")
            else:
                raise ValueError("Unrecognised method.")
            print(inds)
            if inds:
                self.ica.plot_scores(scores, exclude=inds)
                self.comps += inds

    def save(self,comps=None):
        # save the new file
        self.comps += self.ica.exclude
        if not comps:
            self.ica.apply(self.raw,exclude=self.comps).save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)
        elif isinstance(comps,list):
            self.ica.apply(self.raw,exclude=self.comps+comps).save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)
        else:
            print("No components applied, saving anyway for consistency.")
            self.raw.save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)

cyc = Cycler(filelist,ref_comp_num)
