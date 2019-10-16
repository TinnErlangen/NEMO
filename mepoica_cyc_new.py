#step 5 - inspect the ICA results and select components to exclude for all files

import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14"]
#subjs = ["nc_NEM_25"]
runs = ["1","2","3","4"]
#runs=["1","2","3"]

#collecting the files : triplets of annotated epoch file and corresponding reference and MEG ica result files
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(['{dir}{sub}_{run}_m-epo.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-ref-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-meg-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-ica.fif'.format(dir=proc_dir,sub=sub,run=run)])

ref_comp_num = 6   #number of reference components to be used as 'ground'

#definition of cycler object to go through the file list for component selection and exclusion
class Cycler():

    def __init__(self,filelist,ref_comp_num):
        self.filelist = filelist
        self.ref_comp_num = ref_comp_num

        #the go() method plots all components and sources for inspection/selection
    def go(self,idx=0):
        plt.close('all')
        # load the next epo/ICA files
        self.fn = self.filelist.pop(idx)
        self.epo = mne.read_epochs(self.fn[0])
        self.icaref = mne.preprocessing.read_ica(self.fn[1])
        self.icameg = mne.preprocessing.read_ica(self.fn[2])
        self.ica = mne.preprocessing.read_ica(self.fn[3])

        #housekeeping on reference components, add them to raw data
        refcomps = self.icaref.get_sources(self.epo)
        for c in refcomps.ch_names[:self.ref_comp_num]: # they need to have REF_ prefix to be recognised by MNE algorithm
            refcomps.rename_channels({c:"REF_"+c})
        self.epo.add_channels([refcomps])

        self.comps = []

        # plot everything out for overview
        self.ica.plot_components(picks=list(range(40)))
        self.ica.plot_sources(self.epo, stop=8)
        self.epo.plot(n_channels=64,n_epochs=8,scalings=dict(mag=2e-12,ref_meg=3e-12,misc=10))
        self.epo.plot_psd(fmax=60,average=False,bandwidth=0.8)

    def show_file(self):
        print("Current Epoch File: " + self.fn[0])

    def identify_bad(self,method,threshold=0.5):
        # search for components which correlate with noise
        if isinstance(method,str):
            method = [method]
        elif not isinstance(method,list):
            raise ValueError('"method" must be string or list.')
        for meth in method:
            print(meth)
            if meth == "eog":
                inds, scores = self.ica.find_bads_eog(self.epo)
            elif meth == "ecg":
                inds, scores = self.ica.find_bads_ecg(self.epo)
            elif meth == "ref":
                inds, scores = self.ica.find_bads_ref(self.epo, method="separate",
                                                      threshold=threshold,
                                                      bad_measure="cor")
            else:
                raise ValueError("Unrecognised method.")
            print(inds)
            if inds:
                self.ica.plot_scores(scores, exclude=inds)
                self.comps += inds

    def plot_props(self,props=None):
        # in case you want to take a closer look at a component
        if not props:
            props = self.comps
        self.ica.plot_properties(self.epo,props)

    def without(self,comps=None,fmax=60):
        # see what the data would look like if we took comps out
        self.comps += self.ica.exclude
        if not comps:
            comps = self.comps
        test = self.epo.copy()
        test.load_data()
        test = self.ica.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax,average=False,bandwidth=0.8)
        test.plot(n_epochs=8,n_channels=64,scalings=dict(mag=2e-12,ref_meg=3e-12,misc=10))
        self.test = test

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

    def fincheck(self):
        self.epo.plot(n_channels=64,n_epochs=8,scalings=dict(mag=2e-12,ref_meg=3e-12,misc=10))
        self.epo.plot_psd(fmax=50,average=False)

    def finsave(self):
        self.epo.save(self.fn[0][:-10]+'_ica-epo.fif')
        if self.epo.info["bads"]:
            with open(self.fn[0][:-10]+'_badchans.txt', "w") as file:
                file.write("after ICA\n")
                for b in self.epo.info["bads"]:
                    file.write(b+"\n")
        with open(self.fn[0][:-10]+'_epodrops.txt', "w") as file:
            file.write("after ICA\n")
            for inx,d in enumerate(self.epo.drop_log):
                if d == ['USER']:
                    file.write("Epoch No. {inx} Condition {trig}\n".format(inx=inx+1,trig=self.epo.events[inx, 2]))

cyc = Cycler(filelist, ref_comp_num)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file pair in the list for component inspection and selection
#and cyc.save() with "comps =" for the ones to be excluded when done -> it will save the 'cleaned' epochs in a new file
#then cyc.go() goes on to the next file pair again... until list is empty
