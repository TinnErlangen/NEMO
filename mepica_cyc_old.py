#step 5 - inspect the ICA results and select components to exclude for all files

import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_10"]
runs = ["1","2","3","4"]
runs=["4"]

#collecting the files : triplets of annotated epoch file and corresponding reference and MEG ica result files
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(['{dir}{sub}_{run}_m-epo.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-ref-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-meg-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-all-ica.fif'.format(dir=proc_dir,sub=sub,run=run)])

ref_comp_num = 20   #number of reference components to be used as 'ground'

#definition of cycler object to go through the file list for component selection and exclusion
class Cycler():

    def __init__(self,filelist,ref_comp_num):
        self.filelist = filelist
        self.ref_comp_num = ref_comp_num

        #the go() method plots all components and sources for inspection/selection
    def go(self):
        plt.close('all')
        # load the next epo/ICA files
        self.fn = self.filelist.pop()
        self.epo = mne.read_epochs(self.fn[0])
        self.icaref = mne.preprocessing.read_ica(self.fn[1])
        self.icameg = mne.preprocessing.read_ica(self.fn[2])
        self.ica = mne.preprocessing.read_ica(self.fn[3])

        # housekeeping on reference components, add them to raw data
        refcomps = self.icaref.get_sources(self.epo)
        for c in refcomps.ch_names[:self.ref_comp_num]: # they need to have REF_ prefix to be recognised by MNE algorithm
            refcomps.rename_channels({c:"REF_"+c})
        self.epo.add_channels([refcomps])
        self.comps = []

        # plot everything out for overview
        self.icaref.plot_sources(self.epo, picks = list(range(self.ref_comp_num)),stop=10)
        self.icameg.plot_components(picks=list(range(20)))
        self.ica.plot_components(picks=list(range(20)))
        self.icameg.plot_sources(self.epo,stop=10)
        self.ica.plot_sources(self.epo, stop=10)
        self.epo.plot(n_channels=64,n_epochs=8,scalings=dict(mag=2e-12,ref_meg=3e-12,misc=10))
        self.epo.plot_psd(fmax=40,average=False)

    def plot_props(self,props=None,meg=False):
        # in case you want to take a closer look at a component
        if not props:
            props = self.comps
        if meg:
            self.icameg.plot_properties(self.epo,props)
        else:
            self.ica.plot_properties(self.epo,props)

    def show_file(self):
        print("Current Epoch File: " + self.fn[0])

    def without(self,comps=None,fmax=40):
        # see what the data would look like if we took comps out
        if not comps:
            comps = self.comps
        test = self.epo.copy()
        test.load_data()
        test = self.icameg.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax)
        test.plot(n_epochs=8,n_channels=30,scalings=dict(mag=2e-12,ref_meg=3e-12,misc=10))
        self.test = test

    def identify_ref(self,threshold=4):
        # search for components which correlate with reference components
        ref_inds, scores = self.icameg.find_bads_ref(self.epo,threshold=threshold)
        if ref_inds:
            self.icameg.plot_scores(scores, exclude=ref_inds)
            print(ref_inds)
            #self.icameg.plot_properties(self.raw,ref_inds)
            self.comps += ref_inds

    def identify_bad(self,method,threshold=3):
        # search for components which correlate with noise
        if isinstance(method,str):
            method = [method]
        elif not isinstance(method,list):
            raise ValueError('"method" must be string or list.')
        kwrgs = {}
        for meth in method:
            print(meth)
            if meth == "eog":
                func = self.ica.find_bads_eog
            elif meth == "ecg":
                func = self.ica.find_bads_ecg
            elif meth == "ref_t":
                func = self.ica.find_bads_ref
                kwrgs["method"] = "together"
                kwrgs["threshold"] = 3
            elif meth == "ref_s":
                func = self.ica.find_bads_ref
                kwrgs["method"] = "separate"
                kwrgs["threshold"] = 3
            else:
                raise ValueError("Unrecognised method.")
            inds, scores = func(self.epo, **kwrgs)
            print(inds)
            if inds:
                if meth == "ref_s":
                    self.icameg.plot_scores(scores, exclude=inds)
                else:
                    self.ica.plot_scores(scores, exclude=inds)
                self.comps += inds

        #when saving, enter the MEG components to be excluded, bad reference components are excluded automatically
    def save(self,comps=None):
        self.comps += self.ica.exclude
        if not comps:
            self.ica.apply(self.epo,exclude=self.comps).save(self.fn[0][:-10]+'_ica-epo.fif')
        elif isinstance(comps,list):
            self.ica.apply(self.epo,exclude=self.comps+comps).save(self.fn[0][:-10]+'_ica-epo.fif')
        else:
            print("No components applied, saving anyway for consistency")
            self.epo.save(self.fn[0][:-10]+'_ica-epo.fif')

cyc = Cycler(filelist, ref_comp_num)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file pair in the list for component inspection and selection
#and cyc.save() with "comps =" for the ones to be excluded when done -> it will save the 'cleaned' epochs in a new file
#then cyc.go() goes on to the next file pair again... until list is empty
