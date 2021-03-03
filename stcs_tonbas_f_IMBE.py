import mne
import numpy as np

## remember: BRA52, FAO18, WKI71 have fsaverage MRIs (originals were defective)

meg_dir = "D:/NEMO_analyses/proc/"
trans_dir = "D:/NEMO_analyses/proc/trans_files/"
mri_dir = "D:/freesurfer/subjects/"
save_dir = "V:/Alle/Müller-Voggel/TONBAS_files/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}

# freq bands from DICS calcs that went into STCs
freqencies = {"theta":list(np.arange(4,7)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(17,24)),
         "beta_high":list(np.arange(26,35)),"gamma":(np.arange(35,56))}
cycles = {"theta":5,"alpha":10,"beta_low":20,"beta_high":30,"gamma":35}
g_freqs = {"gamma_high":(np.arange(65,96))}
g_cycles = {"gamma_high":35}

# needed here
freqs = ["alpha","gamma","gamma_high"]
stcs = ["stc_restbas","stc_tonb"]

# load existing stcs
for meg, mri in sub_dict.items():

    # DONE.
    # # re-create ton stcs (pre diff calc) for alpha and gamma (gamma_high exists already)
    # # load alpha and gamma restbas stc
    # stc_rest_alpha = mne.read_source_estimate("{}nc_{}_stc_restbas_alpha".format(meg_dir,meg))
    # stc_rest_gamma = mne.read_source_estimate("{}nc_{}_stc_restbas_gamma".format(meg_dir,meg))
    # # load alpha and gamma tonbas stc (this was calculated from "(ton-rest)/rest")
    # stc_tondiff_alpha = mne.read_source_estimate("{}nc_{}_stc_tonbas_alpha".format(meg_dir,meg))
    # stc_tondiff_gamma = mne.read_source_estimate("{}nc_{}_stc_tonbas_gamma".format(meg_dir,meg))
    # # mult and add rest again to get tonb & save to have it
    # stc_ton_alpha = stc_tondiff_alpha * stc_rest_alpha + stc_rest_alpha
    # stc_ton_gamma = stc_tondiff_gamma * stc_rest_gamma + stc_rest_gamma
    # stc_ton_alpha.save("{}nc_{}_stc_tonb_alpha".format(meg_dir,meg))
    # stc_ton_gamma.save("{}nc_{}_stc_tonb_gamma".format(meg_dir,meg))

    # DONE.
    # # loop through all stcs, morph to fsavg and save in D: as well as V:
    # for freq in freqs:
    #     for st_con in stcs:
    #         stc = mne.read_source_estimate("{d}nc_{s}_{sn}_{f}".format(d=meg_dir,s=meg,sn=st_con,f=freq))
    #         morph = mne.compute_source_morph(stc,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir)
    #         stc = morph.apply(stc)
    #         if "rest" in st_con:
    #             stc.save("{d}nc_{s}_fsavg_stc_restbas_{f}".format(d=meg_dir,s=meg,f=freq))
    #             stc.save("{d}nc_{s}_fsavg_stc_rest_{f}".format(d=save_dir,s=meg,f=freq))
    #         elif "ton" in st_con:
    #             stc.save("{d}nc_{s}_fsavg_stc_tonb_{f}".format(d=meg_dir,s=meg,f=freq))
    #             stc.save("{d}nc_{s}_fsavg_stc_ton_{f}".format(d=save_dir,s=meg,f=freq))

    # re-do morph to fsavg with less vertices and save only to V:
    for freq in freqs:
        for st_con in stcs:
            stc = mne.read_source_estimate("{d}nc_{s}_{sn}_{f}".format(d=meg_dir,s=meg,sn=st_con,f=freq))
            # fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(meg_dir))
            morph3 = mne.compute_source_morph(stc,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir,spacing=3)
            morph4 = mne.compute_source_morph(stc,subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir,spacing=4)
            stc3 = morph3.apply(stc)
            stc4 = morph4.apply(stc)
            if "rest" in st_con:
                stc3.save("{d}nc_{s}_fsavg_stc_rest_{f}_klein".format(d=save_dir,s=meg,f=freq))
                stc4.save("{d}nc_{s}_fsavg_stc_rest_{f}_mittel".format(d=save_dir,s=meg,f=freq))
            elif "ton" in st_con:
                stc3.save("{d}nc_{s}_fsavg_stc_ton_{f}_klein".format(d=save_dir,s=meg,f=freq))
                stc4.save("{d}nc_{s}_fsavg_stc_ton_{f}_mittel".format(d=save_dir,s=meg,f=freq))
