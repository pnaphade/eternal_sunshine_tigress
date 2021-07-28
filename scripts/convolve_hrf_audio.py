import numpy as np
from scipy.stats import gamma
from scipy.special import gamma as fgamma

datadir = '/Users/jamalw/Dropbox/music_event_structures/music_features/'
savedir = '/Users/jamalw/Dropbox/music_event_structures/full_regressors/'

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
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf

hrf = single_gamma_hrf(TR = 1)
n_acq = 2511

chromaRun1_hrf = np.apply_along_axis(np.convolve, 1, chromaRun1, hrf, 'full')[:n_acq][:,0:2511]
#mfccRun1_hrf = np.apply_along_axis(np.convolve, 1, mfccRun1, hrf, 'full')[:n_acq][:,0:2511]
#tempoRun1_hrf = np.apply_along_axis(np.convolve, 1, tempoRun1, hrf, 'full')[:n_acq][:,0:2511]
spectRun1_hrf = np.apply_along_axis(np.convolve, 1, spectRun1, hrf, 'full')[:n_acq][:,0:2511]


np.save(savedir + 'chromaRun1_no_hrf',chromaRun1_hrf)
#np.save(savedir + 'mfccRun1_no_hrf', mfccRun1_hrf)
#np.save(savedir + 'tempoRun1_12PC_singles_no_hrf', tempoRun1_hrf)
np.save(savedir + 'spectRun1_12PC_no_hrf', spectRun1)

