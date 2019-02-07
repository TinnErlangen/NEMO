#step 5 - inspect the ICA results and select components to exclude for all files

import mne
import matplotlib.pyplot as plt

plt.ion() #this keeps plots interactive

base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/"
proc_dir = base_dir+"proc/"
subjs = ["nc_NEM_12"]
runs = ["1","2","3","4"]
runs=["1"]

#collecting the files : pairs of annotated epoch file and corresponding ica result file
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(['{dir}{sub}_{run}_m-epo.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_mepo-ica.fif'.format(dir=proc_dir,sub=sub,run=run)])

#definition of cycler object to go through the file list for component selection and exclusion
class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

        #the go() method plots all components and sources for inspection/selection
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

        #when saving, enter the components to be excluded
    def save(self,comps=None):
        if comps:
            self.ica.apply(self.epo,exclude=comps).save(proc_dir+sub+"_"+run+"_ica-epo.fif")
        else:
            print("No components applied, saving anyway for consistency")
            self.epo.save(proc_dir+sub+"_"+run+"_ica-epo.fif")

cyc = Cycler(filelist)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file pair in the list for component inspection and selection
#and cyc.save() with "comps =" for the ones to be excluded when done -> it will save the 'cleaned' epochs in a new file
#then cyc.go() goes on to the next file pair again... until list is empty
