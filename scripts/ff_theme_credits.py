import numpy as np
import matplotlib.pyplot as plt

# Load in the data
data_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
A1_corrs = np.load(data_dir + "hrf_A1_full_length_slide_corrs.npy")
bs_corrs = np.load(data_dir + "hrf_bs_full_length_slide_corrs.npy")
dmn_corrs = np.load(data_dir + "hrf_dmnA_full_length_slide_corrs_NO_RESAMPLE.npy")

# Pull out the chromagram row
A1_corrs = A1_corrs[2, :, :]
bs_corrs = bs_corrs[2, :, :]
dmn_corrs = dmn_corrs[2, :, :]

# Music/no music correlations over focus feature theme (rA1, chroma)
fig1, ax = plt.subplots()
ax.plot(A1_corrs[1, 0:13], 'b', label="Music") # music rA1
ax.plot(A1_corrs[3, 0:13], 'r', label="No Music") # no music rA1
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()
#fig1.show()

# rA1, brainstem, occipital pole correlations over focus feature theme
fig2, ax = plt.subplots()
ax.plot(A1_corrs[1, 0:13],'b', label="Music rA1")
ax.plot(bs_corrs[0, 0:13],'r', label="Music Brainstem")
ax.plot(bs_corrs[1, 0:13], 'g', label="Music Occcipital Pole")
ax.plot(dmn_corrs[0, 0:13], 'm', label="Music DMNa")
plt.ylim(0, 0.7)
ax.set_xlabel("TR")
ax.set_ylabel("Correlation")
ax.legend()
#fig2.show()

# Music/no music correlations over credits
fig3, ax = plt.subplots()
ax.plot(A1_corrs[1, -204:], 'b', label="Music rA1")
ax.plot(bs_corrs[0, -204:], 'r', label="Music Brainstem")
ax.plot(bs_corrs[1, -204:], 'g', label="Music Occipital Pole")
ax.plot(dmn_corrs[0, -204:], 'm', label="Music DMNa")
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()
#fig3.show()
