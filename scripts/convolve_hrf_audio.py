import numpy as np
from scipy.stats import gamma
from scipy.special import gamma as fgamma
import re
from pathlib import Path
import os
import glob


# Load in the audio features, transpose in preparation for correlation
feat_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
feat_paths = glob.glob(os.path.join(datadir, "es*"))
features = [np.load(path) for path in feat_paths]

# Pull out the labels of each feature from their filepaths using regex
feat_labels = [re.search('RSA/(.+?).npy', path).group(1) for path in feat_paths]

# Directory for saving hrf-convolved features
save_dir = feat_dir


def single_gamma_hrf(TR, t=5, d=5.2, onset=0, kernel=32):
    """Single gamma hemodynamic response function.
    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t : float (default=5.4)
        Delay of response relative to onset (in seconds).
    d : float (default=5.2)
        Dispersion of response.
    onset : float (default=0)
        Onset of hemodynamic response (in seconds).
    kernel : float (default=32)
        Length of kernel (in seconds).
    Returns
    -------
    hrf : array
        Hemodynamic repsonse function
    References
    ----------
    [1] Adapted from the pymvpa tools.
        https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/misc/fx.py
    """

    ## Define metadata.
    fMRI_T = 16.0
    TR = float(TR)

    ## Define times.
    dt = TR/fMRI_T
    u  = np.arange(kernel/dt + 1) - onset/dt
    u *= dt

    ## Generate (super-sampled) HRF.
    hrf = (u / t) ** ((t ** 2) / (d ** 2) * 8.0 * np.log(2.0)) \
          * np.e ** ((u - t) / -((d ** 2) / t / 8.0 / np.log(2.0)))

    ## Downsample.
    good_pts=np.array(range(int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf


# Get the hrf
hrf = single_gamma_hrf(TR = 1)

# Number of trs
n_acq = features[0].shape[1]


# Convolve the features and the hrf
hrf_features = []

for feature in features : 

	hrf_feature = np.apply_along_axis(np.convolve, 1, feature, hrf)[:, 0:n_acq]
	
	hrf_features.append(hrf_feature)

#chromaRun1_hrf = np.apply_along_axis(np.convolve, 1, chromaRun1, hrf, 'full')[:n_acq][:,0:2511]


# Save the hrf-convolved features
for hrf_feature, label in zip(hrf_features, feat_labels) :
	
	np.save(save_dir + label + "_hrf.npy", hrf_feature)
