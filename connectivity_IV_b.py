import mne
import numpy as np
import conpy
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mayavi import mlab
from mne.externals.h5io import write_hdf5

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective); WKI71_fa MRI is also blurry ?!

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
save_dir = "D:/NEMO_analyses/plots/connectivity_sum/"   # for plots
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_26":"ENR41"}
freqs = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56)),"gamma_high":(np.arange(65,96))}
# freqs = {"gamma_high":(np.arange(65,96))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35,"gamma_high":35}

exp_conds = ["exp","ton","neg","pos"]
exp_conds = ["neg","ton"]
bas_conds = ["restbas","tonbas"]
contrasts = ["tonbas_vs_rest","neg_vs_pos","neg_vs_ton"]
# contrasts = ["neg_vs_ton"]

# load the fsaverage ico4 source space to morph back to
fs_src = mne.read_source_spaces("{}fsaverage_ico4-src.fif".format(meg_dir))

for contrast in contrasts:
    # for each frequency band:
    for freq,vals in freqs.items():

        # read the statistically pruned connectivity & calculate parcellations (different summary versions)
        con_clust = conpy.read_connectivity('{dir}NEMO_{cont}_contrast_{f}-pruned-avg-connectivity.h5'.format(dir=meg_dir,cont=contrast,f=freq))
        if not con_clust.n_connections == 0:
            # Summarize the connectivity in parcels
            labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s',subjects_dir=mri_dir)
            del labels[-2:]  # drop 'unknown' labels
            con_parc_sum = con_clust.parcellate(labels, summary='sum',weight_by_degree=True)
            con_parc_sum.save('{dir}NEMO_{cont}_contrast_{f}-pruned-label-sum-connectivity.h5'.format(dir=meg_dir,cont=contrast,f=freq))
            # print them out
            print("The following connections by sum remain for  ",contrast,freq)
            print(con_clust)
            print(con_parc_sum)

            # plot and save some of them

            # Plot the sum map with optimized scale
            stc = con_clust.make_stc(summary='sum', weight_by_degree=True)
            stc.data = stc.data * 1000
            fmax = np.abs(stc.data).max()
            fmin = np.abs(stc.data[stc.data!=0]).min()
            fmid = (fmax + fmin)/2
            fig = mlab.figure(size=(300, 300))
            brain = stc.plot(
                subject='fsaverage',
                hemi='both',
                surface='inflated',
                background='white',
                foreground='black',
                time_label='',
                initial_time=0,
                smoothing_steps=5,
                figure=fig,
                colormap ="jet",     # others "bwr"
                subjects_dir=mri_dir,
            )
            brain.scale_data_colormap(fmin, fmid/3, fmax, True,center=0)   # optimize these values !!
            brain.add_annotation('aparc.a2009s', borders=1, alpha=0.5)

            # Save some views
            mlab.view(0, 90, 450, [0, 0, 0])
            mlab.savefig('{d}{cont}_connectivity_{f}_sum_rh.png'.format(d=save_dir,cont=contrast,f=freq), magnification=4)
            mlab.view(180, 90, 450, [0, 0, 0])
            mlab.savefig('{d}{cont}_connectivity_{f}_sum_lh.png'.format(d=save_dir,cont=contrast,f=freq), magnification=4)
            mlab.view(180, 0, 450, [0, 10, 0])
            mlab.savefig('{d}{cont}_connectivity_{f}_sum_top.png'.format(d=save_dir,cont=contrast,f=freq), magnification=4)
            mlab.view(180, 180, 480, [0, 10, 0])
            mlab.savefig('{d}{cont}_connectivity_{f}_sum_bottom.png'.format(d=save_dir,cont=contrast,f=freq), magnification=4)
            mlab.close(fig)

            # Plot the connectivity diagram with optimized diverging colorspace
            con_parc_sum.data = con_parc_sum.data * 1000
            vmax = con_parc_sum.data.max()
            vmin = con_parc_sum.data.min()
            num_levels = 40
            midpoint = 0
            levels = np.linspace(vmin, vmax, num_levels)
            midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
            vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
            colors = plt.cm.seismic(vals)
            cmap, norm = from_levels_and_colors(levels, colors)
            fig, _ = con_parc_sum.plot(title='Parcel-wise Connectivity by Sum', facecolor='white',
                                       textcolor='black', node_edgecolor='white',
                                       colormap=cmap,show=True)
            fig.savefig('{d}{cont}_connectivity_{f}_sum_squircle.pdf'.format(d=save_dir,cont=contrast,f=freq), bbox_inches='tight')
