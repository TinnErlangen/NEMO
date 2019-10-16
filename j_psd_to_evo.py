import numpy as np
from mne import EvokedArray, pick_types
from mne.channels.layout import find_layout
from mne.io.bti.bti import _rename_channels

def psd_to_evo(psd,freqs,epo,keep_ch_names=True):
    if len(psd.shape) == 3:
        psd = np.mean(psd,axis=0)
    epo.pick_types(meg=True)
    evo = epo.average()
    evopsd = EvokedArray(psd,evo.info)
    evopsd.times = freqs
    if not keep_ch_names:
        neuromag_names = _rename_channels(evopsd.ch_names)
        evopsd.rename_channels({old: new for (old,new) in zip(evopsd.ch_names,neuromag_names)})
    return evopsd

def fix_layout(inst):
    layout = find_layout(inst.info)
    layout.names = inst.pick_types(meg=True).ch_names
    return layout
