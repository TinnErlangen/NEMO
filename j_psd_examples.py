'''examples of converting epoched data to PSD, then PSD to "Evoked,"
so they can be easily displayed and manipulated'''

from mne import combine_evoked
from psd_to_evo import psd_to_evo, fix_layout
from mne.time_frequency import psd_welch
from mne.viz import plot_evoked_topo

# these two lines make plotting go more smoothly
import matplotlib.pyplot as plt
plt.ion()

# assume we have two Epochs objects already, stored as variables epo1 and epo2

# have used Welch here, but any other PSD technique will do
psd1, freqs1 = psd_welch(epo1)
psd2, freqs2 = psd_welch(epo2)

# psd and freqs are raw numpy arrays; convert to mne.Evoked objects
psd_evo1 = psd_to_evo(psd1, freqs1, epo1, keep_ch_names=False)
psd_evo2 = psd_to_evo(psd2, freqs2, epo2, keep_ch_names=False)

# Not necessary, but you can name them. Useful for plotting
psd_evo1.comment = "First"
psd_evo2.comment = "Second"

# Any Evoked can be plotted e.g. as a topo.
# as far as MNE knows, this is an ERP, so the X and Y axis
# will not be correctly titled, i.e. the X axis will say it's
# time, when it's actually frequency. Just ignore for now. The
# values themselves are correct.
psd_evo1.plot_topo()

# plot multiple
plot_evoked_topo([psd_evo1, psd_evo2])

# subtract condition two from one
# note this function is very general, and can perform any
# linear operation on any set of Evokeds
one_minus_two = combine_evoked([psd_evo1, psd_evo2], [1, -1])
one_minus_two.comment = "one - two"
one_minus_two.plot_topo()

# Note that the channels names have the generic, Neuromag names, i.e.
# "MEGXXX." The automatic channel layout identifier gets
# confused by the native "AXXX" Magnes3600 names, and so functions such as
# plot_topo fail, because they cannot figure out the physical locations
# of the channels. One way to solve this problem is to convert the channel
# names to Neuromag names with psd_to_evo(keep_ch_names=False). This is
# the approach used above. Alternatively, if you want to keep the Magnes
# names, then you need to specify the layout manually, like so:

psd_evo1 = psd_to_evo(psd1, freqs1, epo1)
psd_evo1.comment = "First"
layout = fix_layout(psd_evo1)
# all functions which involve plotting topographies would need to have
# this layout specificed manually, see line below
psd_evo1.plot_topo(layout=layout)
