# encoding: utf-8
"""
Makes a plot of the grand-average connectivity, contrasted between conditions.
Both the degree map and the circle diagram are plotted. Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import conpy
from mayavi import mlab

meg_dir = "D:/NEMO_analyses/proc/"
mri_dir = "D:/freesurfer/subjects/"
save_dir = "D:/NEMO_analyses/plots/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_26":"ENR41"}

freqs = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
freq = "gamma_high"

# Read the pruned connectivity estimates
con = conpy.read_connectivity('{dir}NEMO_tonbas_vs_rest_contrast_{f}-pruned-avg-connectivity.h5'.format(dir=meg_dir,f=freq))
con_parc = conpy.read_connectivity('{dir}NEMO_tonbas_vs_rest_contrast_{f}-pruned-label-avg-connectivity.h5'.format(dir=meg_dir,f=freq))

# Plot the degree map
stc = con.make_stc(summary='degree', weight_by_degree=False)
fig = mlab.figure(size=(300, 300))
brain = stc.plot(
    subject='fsaverage',
    hemi='both',
    background='white',
    foreground='black',
    time_label='',
    initial_time=0,
    smoothing_steps=5,
    figure=fig,
    subjects_dir=mri_dir,
)
brain.scale_data_colormap(0, 1, stc.data.max(), True)
brain.add_annotation('aparc.a2009s', borders=2)

# Save some views
mlab.view(0, 90, 450, [0, 0, 0])
mlab.savefig('{d}tonbas_connectivity_{f}_degree_rh.png'.format(d=save_dir,f=freq), magnification=4)
mlab.view(180, 90, 450, [0, 0, 0])
mlab.savefig('{d}tonbas_connectivity_{f}_degree_lh.png'.format(d=save_dir,f=freq), magnification=4)
mlab.view(180, 0, 450, [0, 10, 0])
mlab.savefig('{d}tonbas_connectivity_{f}_degree_top.png'.format(d=save_dir,f=freq), magnification=4)
mlab.view(180, 180, 480, [0, 10, 0])
mlab.savefig('{d}tonbas_connectivity_{f}_degree_bottom.png'.format(d=save_dir,f=freq), magnification=4)

# Plot the connectivity diagram
fig, _ = con_parc.plot(title='Parcel-wise Connectivity', facecolor='white',
                       textcolor='black', node_edgecolor='white',
                       colormap='plasma_r', vmin=0, show=True)
fig.savefig('{d}tonbas_connectivity_{f}_squircle.pdf'.format(d=save_dir,f=freq), bbox_inches='tight')
