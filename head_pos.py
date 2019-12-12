import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

proc_dir = "D:/NEMO_analyses/proc/"
subjs = ["nc_NEM_10","nc_NEM_11","nc_NEM_12","nc_NEM_14","nc_NEM_15","nc_NEM_16",
        "nc_NEM_17","nc_NEM_18","nc_NEM_19","nc_NEM_20","nc_NEM_21","nc_NEM_22",
        "nc_NEM_23","nc_NEM_24","nc_NEM_26","nc_NEM_27","nc_NEM_28",
        "nc_NEM_29","nc_NEM_30","nc_NEM_31","nc_NEM_32","nc_NEM_33","nc_NEM_34",
        "nc_NEM_35","nc_NEM_36","nc_NEM_37"]
#subjs = ["ATT_10", "ATT_11", "ATT_12"]
runs = ["1","2","3","4"]
plot = True

colors = [(1,0,0),(0,1,1),(0,1,0),(0,0,1)]

pos = np.zeros((len(subjs),len(runs),3))
plane_norms = np.zeros((len(subjs),len(runs),3))
dist_mat = np.zeros((len(subjs),len(runs),len(runs)))
cos_mat = np.zeros((len(subjs),len(runs),len(runs)))
for sub_idx,sub in enumerate(subjs):
    if plot:
        fig = plt.figure()
        pos_ax = fig.add_subplot(1,2,1,projection="3d")
        pos_ax.set_xlim((-0.005,0.005))
        pos_ax.set_ylim((-0.005,0.005))
        pos_ax.set_zlim((-0.005,0.005))
        rot_ax = fig.add_subplot(1,2,2,projection="3d")
        rot_ax.set_xlim((-1,1))
        rot_ax.set_ylim((-1,1))
        rot_ax.set_zlim((-1,1))
    for run_idx,run in enumerate(runs):
        epo_name = "{dir}{sub}_{run}_ica-epo.fif".format(dir=proc_dir,
                                                             sub=sub, run=run)
        epo = mne.read_epochs(epo_name)
        dev_head_t = epo.info["dev_head_t"]
        head_dev_t = mne.transforms.invert_transform(dev_head_t)
        # distance
        pos[sub_idx,run_idx] = mne.transforms.apply_trans(dev_head_t,
                                                          np.array([0,0,0]))
        # angle
        fid_points = np.array([epo.info["dig"][idx]["r"] for idx in range(3)])
        fid_points_dev = mne.transforms.apply_trans(head_dev_t,fid_points)
        plane_norms[sub_idx,run_idx] = np.cross(fid_points_dev[0,]-fid_points_dev[1,],
                                                fid_points_dev[2,]-fid_points_dev[1,])
        plane_norms[sub_idx,run_idx] /= np.linalg.norm(plane_norms[sub_idx,run_idx])
        if plot:
            rot_ax.quiver(0,0,0,*plane_norms[sub_idx,run_idx],alpha=0.2,
                          color=colors[run_idx])
    if plot:
        pos_centred = pos[sub_idx,] - np.mean(pos[sub_idx,],axis=0)
        pos_ax.scatter(pos_centred[:,0],pos_centred[:,1],zs=pos_centred[:,2],
                       c=colors,alpha=0.2)

    dist_mat[sub_idx,] = distance_matrix(pos[sub_idx,],pos[sub_idx,])
    cos_mat[sub_idx,] = cosine_similarity(plane_norms[sub_idx,])
    if np.max(dist_mat[sub_idx,]>0.005):
        print("Warning: Subject {} produced a distance of more than 5mm".
              format(sub,run))
plt.show()
